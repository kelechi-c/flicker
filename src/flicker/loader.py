"""safetensors -> nested pytree, one tensor at a time.

Three rules and no renaming table (`port-plan` §4):

| torch | jax |
|---|---|
| `*.weight` on a `Linear` | `transpose` -> `(in, out)` |
| `*.weight` on a `Conv2d` (4-D) | `transpose(2,3,1,0)` -> `HWIO` |
| `*.scale`, everything else | as-is (QKNorm scales stay **bf16**) |

The tree nests **one** level at the module the consuming function takes as an argument, and keeps
the rest of the torch path dotted: `p["time_in"]["in_layer.weight"]`,
`p["double_blocks"]["img_attn.qkv.weight"]` (leading axis = block index, ready for `lax.scan`).
The key set is asserted in **both** directions — a missing key and an unexpected key are both
hard errors, because a `ShapeDtypeStruct`-template loader silently no-ops otherwise (§9.1).

One optional post-load rewrite: `fold_smooth` replaces the block GEMMs a SmoothQuant mode names
with `nn.Int8Weight` leaves (calibrated off-TPU), and `save_calibration`/`load_calibration` move
the statistics it folds against. The checkpoint's names stay the contract.
"""

import functools
import inspect
import json
import pathlib as pl

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from .config import Klein4B, Qwen3_4B
from .nn import Int8Weight, quantize_weight_smooth, smooth_scales

Array = jax.Array
Tree = dict[str, Array]

# `backend="pread"` makes `safe_open` read lazily from disk instead of mapping the whole file, which
# is the measured 1.24x-RAM path here. It is only accepted from safetensors 0.5 on, and the reference
# images pin 0.4.5 for the BFL checkout, so feature-detect rather than require it.
_PREAD = "backend" in inspect.signature(safe_open).parameters
open_np = functools.partial(
    safe_open, framework="np", **({"backend": "pread"} if _PREAD else {})
)


def header(path: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """`{key: (dtype, shape)}` without reading any tensor data."""
    with safe_open(path, framework="np") as f:
        return {k: (str(f.get_slice(k).get_dtype()), tuple(f.get_slice(k).get_shape())) for k in f.keys()}


def load_flat(path: str, expected: set[str], transpose: tuple[str, ...] = ("weight",),
              ignore: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Stream every expected tensor, transposing per the rules above; assert the key set.

    `safe_open(backend="pread")` + `get_tensor` one at a time is the measured 1.24x-host-RAM
    path — never `load_file`.
    """
    with open_np(path) as f:
        found = set(f.keys()) - set(ignore)
        missing, extra = expected - found, found - expected
        assert not missing, f"{len(missing)} key(s) missing from {path}: {sorted(missing)[:5]}"
        assert not extra, f"{len(extra)} unexpected key(s) in {path}: {sorted(extra)[:5]}"
        return {k: _orient(f.get_tensor(k), k, transpose) for k in sorted(expected)}


def _orient(w: np.ndarray, key: str, transpose: tuple[str, ...]) -> np.ndarray:
    """Transpose to JAX layout: 4-D conv kernels OIHW->HWIO, 2-D linear weights (out,in)->(in,out)."""
    if not key.endswith(transpose):
        return w
    if w.ndim == 4:
        return np.ascontiguousarray(w.transpose(2, 3, 1, 0))
    if w.ndim == 2:
        return np.ascontiguousarray(w.T)
    return w


def nest(flat: Tree) -> Tree:
    """`{"a.b.c": x}` -> `{"a": {"b.c": x}}` — one level of grouping by consuming module."""
    out: Tree = {}
    for key, value in flat.items():
        head, _, rest = key.partition(".")
        out.setdefault(head, {})[rest if rest else "weight"] = value
    return out


def stack_blocks(flat: Tree, prefix: str, count: int) -> Tree:
    """`{prefix.i.rest}` -> `{rest: stacked with leading axis i}` for `lax.scan`."""
    out: Tree = {}
    for index in range(count):
        head = f"{prefix}.{index}."
        for key, value in flat.items():
            if key.startswith(head):
                out.setdefault(key[len(head) :], []).append(value)
    assert out and len(next(iter(out.values()))) == count, f"{prefix}: expected {count} blocks"
    return {key: jnp.stack(values) for key, values in out.items()}


def dit_keys(cfg: Klein4B) -> set[str]:
    """The 149 tensor names of `flux-2-klein-4b.safetensors` (project.md §7 closes exactly)."""
    keys = {
        "img_in.weight",
        "txt_in.weight",
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.lin.weight",
        "single_stream_modulation.lin.weight",
        "final_layer.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    }
    for i in range(cfg.depth):
        for stream in ("img", "txt"):
            keys |= {
                f"double_blocks.{i}.{stream}_attn.qkv.weight",
                f"double_blocks.{i}.{stream}_attn.proj.weight",
                f"double_blocks.{i}.{stream}_attn.norm.query_norm.scale",
                f"double_blocks.{i}.{stream}_attn.norm.key_norm.scale",
                f"double_blocks.{i}.{stream}_mlp.0.weight",
                f"double_blocks.{i}.{stream}_mlp.2.weight",
            }
    for i in range(cfg.depth_single_blocks):
        keys |= {
            f"single_blocks.{i}.linear1.weight",
            f"single_blocks.{i}.linear2.weight",
            f"single_blocks.{i}.norm.query_norm.scale",
            f"single_blocks.{i}.norm.key_norm.scale",
        }
    return keys


# Which block GEMMs each `Klein4B.gemm` mode quantizes, as the *stacked* leaf keys. "smooth_large"
# is the two GEMMs D4 measured a real speedup on: the single-stream `linear1` (unsplit — D4 showed
# the load-time split gains nothing) and the double-stream `img_mlp.0`. Embedders, modulations,
# attention projections and the final layer are never named, so they stay bf16 in every mode.
INT8_MODES: dict[str, dict[str, tuple[str, ...]]] = {
    "bf16": {},
    "smooth_large": {"double_blocks": ("img_mlp.0.weight",), "single_blocks": ("linear1.weight",)},
}
# Modes that need a calibration pass: they cannot be built from the checkpoint alone, so `load_dit`
# refuses them loudly instead of quietly handing back a bf16 tree (the D0 silent no-op).
SMOOTH_MODES = ("smooth_large",)


def load_dit(path: str, cfg: Klein4B) -> Tree:
    """Stream the DiT checkpoint into the scan-ready pytree of device arrays.

    A non-bf16 `cfg.gemm` raises: those leaves need activation statistics, so they are built by
    `fold_smooth` on the tree this returns loaded with `gemm="bf16"`. Without the refusal a
    mis-wired arm would silently measure bf16 twice.
    """
    if cfg.gemm != "bf16":
        raise ValueError(f"gemm={cfg.gemm!r} needs activation statistics; load with gemm='bf16' and "
                         f"fold with fold_smooth")
    flat = load_flat(path, dit_keys(cfg))
    blocks = {
        "double_blocks": stack_blocks(flat, "double_blocks", cfg.depth),
        "single_blocks": stack_blocks(flat, "single_blocks", cfg.depth_single_blocks),
    }
    top = {k: v for k, v in flat.items() if not k.startswith(("double_blocks.", "single_blocks."))}
    return jax.device_put(nest(top) | blocks)


def fold_smooth(tree: Tree, mode: str, absmax: dict[str, Array], alpha: float) -> Tree:
    """SmoothQuant: replace every leaf `mode` names by its `diag(s)`-folded `Int8Weight`. Pure.

    `absmax[f"{stack}.{key}"]` is the calibration's per-input-channel activation absmax for that
    leaf, shape `(depth, in)` — one entry per **block**, which is what makes the fold per-block
    rather than per-GEMM-name. The weight's own per-input-channel absmax is read off the leaf, and
    `nn.smooth_scales` combines the two. The dense tree is untouched, so one load serves every
    alpha and mode and a candidate costs one fold (~half the quantized leaves' bytes).
    """
    if mode not in SMOOTH_MODES:
        raise ValueError(f"{mode!r} is not a smooth mode; expected one of {SMOOTH_MODES}")
    out: Tree = {}
    for stack, keys in INT8_MODES[mode].items():
        if stack not in tree:
            continue
        block: Tree = {}
        for key, value in tree[stack].items():
            if key not in keys:
                block[key] = value
                continue
            channel = absmax.get(f"{stack}.{key}")
            if channel is None:
                raise KeyError(f"no calibration absmax for {stack}.{key}")
            block[key] = _fold_per_block(value, channel, alpha)
        out[stack] = block
    return tree | out


def _fold_per_block(value: Array, channel: Array, alpha: float) -> Int8Weight:
    """`quantize_weight_smooth` one block at a time, then re-stacked.

    The stacked fp32 upcast of `single_blocks.linear1` alone is 6.3 GiB — it OOM'd the D5 session
    with the dense model resident. Folding per block caps the fp32 working set at one layer
    (~340 MB) and gives the same leaf.
    """
    parts = [quantize_weight_smooth(
                 value[i], smooth_scales(channel[i], jnp.max(jnp.abs(value[i].astype(jnp.float32)),
                                                             axis=-1), alpha))
             for i in range(value.shape[0])]
    return Int8Weight(*(jnp.stack(leaves) for leaves in zip(*parts)))


def save_calibration(path: str, absmax: dict[str, Array]) -> None:
    """Write a calibration's taps to one safetensors file: keys `"{stack}.{leaf}"`, `(depth, in)`.

    Fail-closed on the shape contract: `fold_smooth` looks the taps up by leaf name and folds per
    block, so a wrong leading axis would fold the wrong block's statistics into a leaf — legal
    arithmetic and silently meaningless. The key set is therefore checked here, exactly like the
    checkpoint's.
    """
    from safetensors.numpy import save_file

    checks = {f"{stack}.{key}" for stack, keys in INT8_MODES["smooth_large"].items() for key in keys}
    if set(absmax) != checks:
        raise ValueError(f"calibration keys are not the smooth_large leaf set: {sorted(absmax)}")
    save_file({key: np.asarray(value, np.float32) for key, value in sorted(absmax.items())}, path)


def load_calibration(path: str) -> dict[str, Array]:
    """Read a calibration safetensors back into `{stack.leaf: (depth, in)}` fp32 arrays."""
    from safetensors.numpy import load_file

    return {key: jnp.asarray(value) for key, value in load_file(path).items()}


EMBED_KEY = "model.embed_tokens.weight"


def text_keys(cfg: Qwen3_4B) -> set[str]:
    """The 298 tensors of Qwen3 layers 0-26 plus the embedding — layers 27-35 and the LM head
    are never run (we tap `hidden_states[27]`), which is ~25 % of the encoder."""
    keys = {EMBED_KEY}
    for i in range(cfg.layers_needed):
        keys |= {
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        }
    return keys


def load_text(index_path: str, cfg: Qwen3_4B) -> Tree:
    """Stream the two Qwen3 shards through their HF index, keeping only the layers we run.

    Each shard is opened once and read with `pread`; only the keys the index maps to it are
    touched, so the 9 unused layers are never materialised. Returns device arrays.
    """
    weight_map = json.loads(pl.Path(index_path).read_text())["weight_map"]
    expected = text_keys(cfg)
    missing = expected - set(weight_map)
    assert not missing, f"{len(missing)} text key(s) absent from the index: {sorted(missing)[:5]}"
    flat: Tree = {}
    for shard in sorted({weight_map[k] for k in expected}):
        keys = sorted(k for k in expected if weight_map[k] == shard)
        with open_np(str(pl.Path(index_path).parent / shard)) as f:
            # the embedding is a lookup table, not a Linear: it must not be transposed
            flat |= {
                k: f.get_tensor(k) if k == EMBED_KEY else _orient(f.get_tensor(k), k, ("weight",))
                for k in keys
            }
    return jax.device_put({
        "embed_tokens": {"weight": flat[EMBED_KEY]},
        "layers": stack_blocks(flat, "model.layers", cfg.layers_needed),
    })
