"""safetensors -> nested pytree, one tensor at a time.

Three rules and no renaming table:

| torch | jax |
|---|---|
| `*.weight` on a `Linear` | `transpose` -> `(in, out)` |
| `*.weight` on a `Conv2d` (4-D) | `transpose(2,3,1,0)` -> `HWIO` |
| `*.scale`, everything else | as-is (QKNorm scales stay **bf16**) |

The tree nests **one** level at the module the consuming function takes as an argument, and keeps
the rest of the torch path dotted: `p["time_in"]["in_layer.weight"]`,
`p["double_blocks"]["img_attn.qkv.weight"]` (leading axis = block index, ready for `lax.scan`).
The key set is asserted in **both** directions — a missing key and an unexpected key are both
hard errors, because a `ShapeDtypeStruct`-template loader silently no-ops otherwise.
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
from .kernels import EXP_LIMIT, logit_bound

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
        return {k: (str(f.get_slice(k).get_dtype()), tuple(f.get_slice(k).get_shape())) for k in f.keys()}  # noqa: SIM118 (not iterable)


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
    """The 149 tensor names of `flux-2-klein-4b.safetensors`."""
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


def load_dit(path: str, cfg: Klein4B) -> Tree:
    """Stream the DiT checkpoint into the scan-ready pytree of device arrays."""
    flat = load_flat(path, dit_keys(cfg))
    blocks = {
        "double_blocks": stack_blocks(flat, "double_blocks", cfg.depth),
        "single_blocks": stack_blocks(flat, "single_blocks", cfg.depth_single_blocks),
    }
    if cfg.attention == "bounded":
        check_logit_bound(blocks)
    top = {k: v for k, v in flat.items() if not k.startswith(("double_blocks.", "single_blocks."))}
    return jax.device_put(nest(top) | blocks)


def check_logit_bound(blocks: Tree) -> None:
    """Refuse `attention="bounded"` for weights whose QK-norm scales allow `exp` to overflow.

    The double blocks attend over `[txt | img]` jointly, so their bound takes the larger scale of
    the two streams on each side.
    """
    double, single = blocks["double_blocks"], blocks["single_blocks"]

    def joint(name: str) -> np.ndarray:
        return np.maximum(*(np.abs(np.asarray(double[f"{s}_attn.norm.{name}.scale"], np.float32))
                            for s in ("img", "txt")))

    bounds = (logit_bound(joint("query_norm"), joint("key_norm")),
              logit_bound(*(np.asarray(single[f"norm.{n}.scale"], np.float32)
                            for n in ("query_norm", "key_norm"))))
    worst = max(float(np.max(np.asarray(b))) for b in bounds)
    if worst >= EXP_LIMIT:
        raise ValueError(f"attention='bounded' needs logits under {EXP_LIMIT}; these weights allow "
                         f"{worst:.1f}. use attention='flash'")


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
