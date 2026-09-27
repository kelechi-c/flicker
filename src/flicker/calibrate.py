"""Calibration for the SmoothQuant int8 tier: the per-input-channel activation absmax a fold needs.

`absmax` runs a bf16 pass over real conditioning and cached noise and returns, per folded GEMM, the
largest `|input|` of each input channel over the Euler steps and then over the prompts:

    taps = calibrate.absmax(dit, cfg, ctxs, noises)   # {"single_blocks.linear1.weight": (20, 3072),
                                                      #  "double_blocks.img_mlp.0.weight": (5, 12288)}
    loader.save_calibration("calib.safetensors", taps)
    tree = loader.fold_smooth(dit, "smooth_large", taps, alpha=0.35)

The tap forward is `dit.forward`'s arithmetic with the two folded GEMMs' **inputs** emitted as the
block scans' `ys` (the tensors a block computes but does not keep are recomputed from the same
expressions), so the taps are the real GEMM inputs — and the first prompt pins that by checking the
tap forward's velocity against `dit.forward`'s bit for bit. If it differs, the taps are not the real
GEMM inputs and the fold is meaningless, so that mismatch is fatal rather than a warning.
"""

import jax
import jax.numpy as jnp
from jax import lax

from . import modules, rope, sampling
from .config import Config, timesteps
from .dit import forward
from .kernels import Freqs, lane_pair
from .nn import Attention, flash_sdpa, linear, sdpa, timestep_embedding
from .rope import embed_nd

Array = jax.Array
# the leaves this calibration taps, per stack: `loader.INT8_MODES["smooth_large"]`'s leaf set. One
# tap each, at the GEMM input, which is what `fold_smooth` folds the scale against.
TAPS: dict[str, tuple[str, ...]] = {"double_blocks": ("img_mlp.0.weight",),
                                    "single_blocks": ("linear1.weight",)}


def _tap_double(p: dict[str, Array], img: Array, txt: Array, pe_x: Array, pe_ctx: Array,
                mod_img: Array, mod_txt: Array, num_heads: int, attend: Attention,
                freqs: Freqs | None) -> tuple[tuple[Array, Array], dict[str, Array]]:
    """`modules.double_block` with `img_mlp.0`'s input emitted as a tap.

    Arithmetic verbatim: same `double_prepare_qkv`, same residual/gate order, same `silu_gate`. The
    modulated MLP input is the one tensor the block computes but does not keep, so it is named here
    instead of being recovered afterwards.
    """
    state = modules.double_prepare_qkv(p, img, txt, pe_x, pe_ctx, mod_img, mod_txt, num_heads, freqs)
    attn = modules.merge_heads(attend(state.q, state.k, state.v))
    txt_attn, img_attn = attn[:, : state.num_txt], attn[:, state.num_txt:]
    img1 = img + state.img_mod1.gate * linear(img_attn, p["img_attn.proj.weight"])
    mlp_in = modules.modulate(img1, state.img_mod2)
    img2 = img1 + state.img_mod2.gate * linear(
        modules.silu_gate(linear(mlp_in, p["img_mlp.0.weight"])), p["img_mlp.2.weight"]
    )
    txt1 = txt + state.txt_mod1.gate * linear(txt_attn, p["txt_attn.proj.weight"])
    txt2 = txt1 + state.txt_mod2.gate * linear(
        modules.silu_gate(linear(modules.modulate(txt1, state.txt_mod2), p["txt_mlp.0.weight"])),
        p["txt_mlp.2.weight"],
    )
    return (img2, txt2), {"img_mlp.0.weight": jnp.max(jnp.abs(mlp_in), axis=(0, 1))}


def _tap_single(p: dict[str, Array], x: Array, pe: Array, mod: Array, num_heads: int,
                attend: Attention, freqs: Freqs | None) -> tuple[Array, dict[str, Array]]:
    """`modules.single_block` with `linear1`'s input emitted as a tap."""
    state = modules.single_prepare_qkv(p, x, mod, num_heads, pe, freqs)
    attn = modules.merge_heads(attend(state.q, state.k, state.v))
    joined = jnp.concatenate([attn, modules.silu_gate(state.mlp)], axis=-1)
    mlp_in = modules.modulate(x, modules.split_triple(mod, 0))
    return (x + state.gate * linear(joined, p["linear2.weight"]),
            {"linear1.weight": jnp.max(jnp.abs(mlp_in), axis=(0, 1))})


def _tap_forward(p: dict[str, Array], cfg: object, x: Array, x_ids: Array, t: Array, ctx: Array,
                 ctx_ids: Array) -> tuple[Array, dict[str, Array]]:
    """`dit.forward` with the two folded GEMMs' per-input-channel absmax as taps.

    `taps` is keyed `"<stack>.<leaf>"` with shape `(depth, in)` — one entry per **block**, which is
    what makes the fold per-block and not per-GEMM-name. Same ops in the same order as
    `dit.forward`, so the velocity is bit-identical (pinned by `_pin`).
    """
    on_tpu = jax.default_backend() == "tpu"
    attend = flash_sdpa if cfg.attention == "flash" and on_tpu else sdpa
    vec = modules.mlp_embedder(p["time_in"], timestep_embedding(t, 256))
    mod_img = modules.modulation(p["double_stream_modulation_img"], vec, double=True)
    mod_txt = modules.modulation(p["double_stream_modulation_txt"], vec, double=True)
    mod_single = modules.modulation(p["single_stream_modulation"], vec, double=False)
    img, txt = linear(x, p["img_in"]["weight"]), linear(ctx, p["txt_in"]["weight"])
    pe_x = embed_nd(x_ids, cfg.axes_dim, cfg.theta)
    pe_ctx = embed_nd(ctx_ids, cfg.axes_dim, cfg.theta)
    pe = jnp.concatenate([pe_ctx, pe_x], axis=2)
    freqs = lane_pair(pe) if cfg.qk_prep == "kernel" and on_tpu else None

    def one_double(carry: tuple[Array, Array], block: dict[str, Array]
                   ) -> tuple[tuple[Array, Array], dict[str, Array]]:
        return _tap_double(block, *carry, pe_x, pe_ctx, mod_img, mod_txt, cfg.num_heads, attend,
                           freqs)

    (img, txt), double_taps = lax.scan(one_double, (img, txt), p["double_blocks"],
                                       unroll=cfg.block_unroll)
    seq, num_txt = jnp.concatenate([txt, img], axis=1), ctx.shape[1]

    def one_single(x: Array, block: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        return _tap_single(block, x, pe, mod_single, cfg.num_heads, attend, freqs)

    seq, single_taps = lax.scan(one_single, seq, p["single_blocks"], unroll=cfg.block_unroll)
    taps = {f"{stack}.{key}": value for stack, values in
            (("double_blocks", double_taps), ("single_blocks", single_taps))
            for key, value in values.items()}
    return modules.last_layer(p["final_layer"], seq[:, num_txt:], vec), taps


def _tap_denoise(p: dict[str, Array], cfg: object, x: Array, x_ids: Array, ctx: Array,
                 ctx_ids: Array, steps: tuple[float, ...]) -> tuple[Array, dict[str, Array]]:
    """`sampling.denoise`'s Euler scan with the taps as its `ys`: `ys[key]` is `(T, depth, in)`."""
    def step(x: Array, inputs: tuple[Array, Array]) -> tuple[Array, dict[str, Array]]:
        t, delta = inputs
        pred, taps = _tap_forward(p, cfg, x, x_ids, jnp.broadcast_to(t, (x.shape[0],)), ctx,
                                  ctx_ids)
        return x + delta * pred, taps

    return lax.scan(step, x, sampling._schedule(steps))


def _pin(p: dict[str, Array], cfg: object, x: Array, x_ids: Array, ctx: Array, ctx_ids: Array,
         steps: tuple[float, ...]) -> None:
    """The tap forward must be `dit.forward`: check its step-0 velocity, bit for bit."""
    t0 = jnp.broadcast_to(sampling._schedule(steps)[0][0], (x.shape[0],))
    both = jax.jit(lambda p, x, ctx: (_tap_forward(p, cfg, x, x_ids, t0, ctx, ctx_ids)[0],
                                      forward(p, cfg, x, x_ids, t0, ctx, ctx_ids)))
    tapped, reference = both(p, x, ctx)
    if not bool(jnp.array_equal(tapped, reference)):
        raise ValueError("the tap forward's velocity differs from `dit.forward`: the taps are not "
                         "the real GEMM inputs, so a fold against them would be meaningless")


def absmax(dit_params: dict[str, Array], cfg: Config, ctxs: Array, noises: Array
           ) -> dict[str, Array]:
    """Per-input-channel activation absmax for every folded GEMM, from a bf16 tap pass.

    `ctxs` is `(P, S, context_in_dim)` conditioning (one row per calibration prompt, the real
    encoder output or the official ctx), `noises` the cached latent draws `(P, 128, h, w)` the pass
    runs from. The result is `{stack.leaf: (depth, in)}` fp32, max over the Euler steps and then
    over the prompts — the statistics `fold_smooth` folds against, for any alpha.
    """
    side_h, side_w = noises.shape[-2:]
    x_ids = jnp.asarray(rope.img_ids(side_h, side_w))
    ctx_ids = jnp.asarray(rope.txt_ids(ctxs.shape[1]))
    steps = timesteps(cfg.num_steps, side_h * side_w)
    packed = sampling.pack(noises)
    run = jax.jit(lambda p, x, ctx: _tap_denoise(p, cfg.dit, x, x_ids, ctx, ctx_ids, steps))
    total: dict[str, Array] | None = None
    for index in range(packed.shape[0]):
        x, ctx = packed[index: index + 1], ctxs[index: index + 1]
        _, per_step = run(dit_params, x, ctx)
        if index == 0:
            _pin(dit_params, cfg.dit, x, x_ids, ctx, ctx_ids, steps)
        step_max = jax.tree.map(lambda value: jnp.max(value, axis=0), per_step)
        total = step_max if total is None else jax.tree.map(jnp.maximum, total, step_max)
    if total is None:
        raise ValueError("no calibration prompts: `ctxs` and `noises` are empty")
    expected = {f"{stack}.{key}" for stack, keys in TAPS.items() for key in keys}
    if set(total) != expected:
        raise ValueError(f"taps {sorted(total)} are not the smooth_large leaf set {sorted(expected)}")
    # fp32 out: the reduction is over the bf16 pass's own values (`bf16 -> fp32` is exact), so this
    # is the same statistic `save_calibration`/`load_calibration` round trip, not a second one.
    return jax.block_until_ready(jax.tree.map(lambda value: value.astype(jnp.float32), total))
