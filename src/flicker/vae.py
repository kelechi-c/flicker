"""The VAE decoder — diffusers `AutoencoderKLFlux2`, bf16 weights and bf16 activations.

The i2i **encoder** path is out of scope; `load_vae` streams only `decoder.*`, `post_quant_conv` and
the two batch-norm stats (142 of the file's 251 tensors).

Decode chain, exactly as `tools/stage_b_vae.py` captures it (quoted in `tests/golden/vae.json`):

    z (B,128,h,w)  --bn inversion-->  z_in  --unpatchify-->  (B,32,2h,2w)  --post_quant_conv-->
    decoder: conv_in -> mid_block -> up_blocks[0..3] -> conv_norm_out -> silu -> conv_out

The batch-norm inversion is **not** in `AutoencoderKLFlux2.decode` — `Flux2KleinPipeline.__call__`
applies it to the packed latent first, in **bf16**, `sqrt(var + eps)` included. That contradicts
`docs/port-plan.md` decision #5 ("compute it in fp32"); the oracle wins, and the two differ by
3.9e-3 relative on `z_in`. See `inv_normalize`.

Layout: NCHW at the public boundary (the fixture contract is NCHW), NHWC inside, because
`nn.conv2d` is an NHWC `lax.conv_general_dilated`.
"""

import jax
import jax.numpy as jnp

from .config import VAE
from .loader import header, load_flat
from .modules import attn_block, conv, gn, resnet_block, unpatchify, upsample
from .nn import conv2d

Array = jax.Array
Params = dict[str, Array]

DECODER = "decoder"
UP_BLOCKS = "decoder.up_blocks"

# everything outside the decoder path: the encoder side (i2i is out of scope), its `quant_conv`,
# and the training step counter
IGNORED = ("bn.num_batches_tracked",)
ENCODER_PREFIXES = ("encoder.", "quant_conv.")
ENCODER_TENSORS = 108  # 106 encoder + 2 quant_conv; asserted so checkpoint drift is loud


# ---------------------------------------------------------------------------- key set + loader
def _resnet_keys(shortcut: bool) -> set[str]:
    """One `ResnetBlock2D`: two norm/conv pairs, plus the 1x1 skip conv when the width changes."""
    keys = {
        "norm1.weight", "norm1.bias", "conv1.weight", "conv1.bias",
        "norm2.weight", "norm2.bias", "conv2.weight", "conv2.bias",
    }
    if shortcut:
        keys |= {"conv_shortcut.weight", "conv_shortcut.bias"}
    return keys


def _attn_keys() -> set[str]:
    """One `Attention`: GroupNorm, four 1x1 `nn.Linear`s (`to_out` is a 2-element `ModuleList`)."""
    return {
        "group_norm.weight", "group_norm.bias",
        "to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias",
        "to_v.weight", "to_v.bias", "to_out.0.weight", "to_out.0.bias",
    }


def vae_keys(cfg: VAE) -> set[str]:
    """The 142 decoder-path names of `vae/diffusion_pytorch_model.safetensors` (`project.md` §5).

    Up-level widths run backwards over `ch_mult` (`block_out_channels[-1]` first), and a level's
    first resnet carries the 1x1 `conv_shortcut` exactly when the level changes width.
    """
    levels, resnets = len(cfg.ch_mult), cfg.num_res_blocks + 1
    keys = {"bn.running_mean", "bn.running_var", "post_quant_conv.weight", "post_quant_conv.bias"}
    keys |= {f"decoder.{k}" for k in (
        "conv_in.weight", "conv_in.bias", "conv_norm_out.weight", "conv_norm_out.bias",
        "conv_out.weight", "conv_out.bias",
    )}
    keys |= {f"decoder.mid_block.attentions.0.{k}" for k in _attn_keys()}
    keys |= {f"decoder.mid_block.resnets.{j}.{k}" for j in range(2) for k in _resnet_keys(False)}
    out_ch = [cfg.ch * cfg.ch_mult[levels - 1 - i] for i in range(levels)]
    for i in range(levels):
        in_ch = cfg.ch * cfg.ch_mult[-1] if i == 0 else out_ch[i - 1]
        for j in range(resnets):
            width = in_ch if j == 0 else out_ch[i]
            keys |= {f"decoder.up_blocks.{i}.resnets.{j}.{k}" for k in _resnet_keys(width != out_ch[i])}
        if i < levels - 1:
            keys |= {f"decoder.up_blocks.{i}.upsamplers.0.conv.{k}" for k in ("weight", "bias")}
    return keys


def _group(flat: Params, prefix: str) -> Params:
    """`{"decoder.conv_in.weight": x}` -> `{"conv_in.weight": x}` — the torch path minus the prefix."""
    return {k[len(prefix) + 1 :]: v for k, v in flat.items() if k.startswith(prefix + ".")}


def load_vae(path: str, cfg: VAE) -> Params:
    """Stream the checkpoint into the torch-named tree of device arrays, one level of nesting per
    consuming function.

    `p["bn"]["running_var"]`, `p["post_quant_conv"]["weight"]`, `p["decoder"]["conv_in.weight"]`,
    `p["decoder.up_blocks"]["0.resnets.0.conv1.weight"]`. `load_flat` asserts the key set in both
    directions; the 109 encoder-side tensors are *declared* ignorable, and their count is asserted
    so a reshaped checkpoint cannot pass silently.
    """
    ignored = tuple(k for k in header(path) if k.startswith(ENCODER_PREFIXES))
    assert len(ignored) == ENCODER_TENSORS, f"{ENCODER_TENSORS} encoder tensors expected, {len(ignored)} found"
    flat = load_flat(path, vae_keys(cfg), ignore=IGNORED + ignored)
    decoder = {k: v for k, v in _group(flat, DECODER).items() if not k.startswith("up_blocks.")}
    return jax.device_put({
        "bn": _group(flat, "bn"),
        "post_quant_conv": _group(flat, "post_quant_conv"),
        DECODER: decoder,
        UP_BLOCKS: _group(flat, UP_BLOCKS),
    })


# ---------------------------------------------------------------------------- layout helpers
def _nhwc(x: Array) -> Array:
    return jnp.transpose(x, (0, 2, 3, 1))


def _nchw(x: Array) -> Array:
    return jnp.transpose(x, (0, 3, 1, 2))


def _put(trace: dict[str, Array] | None, key: str, x: Array) -> None:
    """Record one contract key (`docs/fixture-contract.md`) — NCHW, as the golden file holds it."""
    if trace is not None:
        trace[key] = x


# ---------------------------------------------------------------------------- the decoder
def decoder_forward(p: Params, x: Array, cfg: VAE, trace: dict[str, Array] | None = None) -> Array:
    """`Decoder.forward` on the unpatchified, post-`post_quant_conv` latent. NCHW in and out.

    `trace` collects `dec/conv_in`, `dec/mid`, `dec/up/{0..3}` (after each level's upsampler, i.e.
    diffusers `up_blocks.i` output) and `dec/out`.
    """
    d, up = p[DECODER], p[UP_BLOCKS]
    h = conv(_nhwc(x), d, "conv_in")
    _put(trace, "dec/conv_in", _nchw(h))
    h = resnet_block(h, d, "mid_block.resnets.0", cfg)
    h = attn_block(h, d, "mid_block.attentions.0", cfg)
    h = resnet_block(h, d, "mid_block.resnets.1", cfg)
    _put(trace, "dec/mid", _nchw(h))
    for i in range(len(cfg.ch_mult)):
        for j in range(cfg.num_res_blocks + 1):
            h = resnet_block(h, up, f"{i}.resnets.{j}", cfg)
        if i < len(cfg.ch_mult) - 1:
            h = upsample(h, up, f"{i}.upsamplers.0", cfg)
        _put(trace, f"dec/up/{i}", _nchw(h))
    h = conv(jax.nn.silu(gn(h, d, "conv_norm_out", cfg)), d, "conv_out")
    _put(trace, "dec/out", _nchw(h))
    return _nchw(h)


def _round_bf16(x: Array) -> Array:
    """Round to bf16 and hand back fp32. Torch eager rounds at every kernel boundary; XLA would
    otherwise fuse `z * std + mean` into a single rounding, which torch measurably does not do."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def inv_normalize(p: Params, z: Array, cfg: VAE) -> Array:
    """`Flux2KleinPipeline`'s batch-norm inversion, in the dtype the oracle uses.

    The stats are bf16 and stay bf16: `var + eps` rounds to bf16, `sqrt` is fp32-then-bf16 (what
    `torch.sqrt` does to a bf16 tensor), and `z * std + mean` is two eager kernels. Computing the
    sqrt in fp32 — `docs/port-plan.md` decision #5, and what `tools/dump_reference.py:379` does —
    moves `z_in` by 3.9e-3 relative, so this function deliberately does not.
    """
    var, mean = (p["bn"][k].astype(jnp.float32).reshape(1, -1, 1, 1)
                 for k in ("running_var", "running_mean"))
    std = _round_bf16(jnp.sqrt(_round_bf16(var + cfg.bn_eps)))
    return (_round_bf16(z.astype(jnp.float32) * std) + mean).astype(z.dtype)


def vae_decode(p: Params, z: Array, cfg: VAE, trace: dict[str, Array] | None = None) -> Array:
    """Packed latent `(B,128,h,w)` -> image `(B,3,8h,8w)`.

    `trace`, when given, is filled with the contract's `vae/` key suffixes: `unpatch`, `z_in`,
    `dec/conv_in`, `dec/mid`, `dec/up/{0..3}`, `dec/out`. The default path is side-effect free.
    """
    _put(trace, "unpatch", z)
    z_in = inv_normalize(p, z, cfg)
    _put(trace, "z_in", z_in)
    pq = p["post_quant_conv"]
    x = conv2d(_nhwc(unpatchify(z_in, cfg.patch)), pq["weight"], padding="VALID") + pq["bias"]
    return decoder_forward(p, _nchw(x), cfg, trace)
