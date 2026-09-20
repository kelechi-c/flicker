"""The layer components, one function per torch module.

Param leaves are the torch `state_dict` names, transposed to JAX layout at load time:
`*.weight` on a `Linear` -> `(in, out)`; `*.scale` (QKNorm) stays `(128,)` and stays **bf16**;
block stacks carry a leading axis so `lax.scan` can walk them.

Everything here is a pure function `(params, activations) -> activations`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Qwen3_4B, VAE
from .nn import (
    conv2d,
    group_norm,
    layer_norm,
    linear,
    rms_norm,
    sdpa,
    silu_gate,
    upsample_nearest,
)
from .rope import apply_rope

Array = jax.Array
Params = dict[str, Array]

# a modulation is always a (B, k, 1, H) tensor: k = 6 (double: two triples) or 3 (single).
Mod = Array


class ModTriple(NamedTuple):
    """`(shift, scale, gate)` — the order `Modulation.forward` chunks in (`model.py:411`)."""

    shift: Array
    scale: Array
    gate: Array


def split_triple(mod: Array, index: int) -> ModTriple:
    """Slice one `(shift, scale, gate)` triple out of a `(B, k, 1, H)` modulation tensor."""
    return ModTriple(mod[:, index], mod[:, index + 1], mod[:, index + 2])


def modulate(x: Array, mod: ModTriple, norm: bool = True) -> Array:
    """`x_mod = (1 + scale) * norm(x) + shift` — note the `1 +`, it is not a plain affine."""
    base = layer_norm(x) if norm else x
    return (1 + mod.scale) * base + mod.shift


def modulation(p: dict[str, Array], vec: Array, double: bool) -> Array:
    """`Modulation` (`model.py:400-412`): `Linear(SiLU(vec))` -> `(B, 6|3, 1, H)`.

    Returns the un-split tensor; `split_triple` indexes it. Shared by all 25 blocks.
    """
    out = linear(jax.nn.silu(vec), p["lin.weight"])
    return out[:, None, :].reshape(out.shape[0], -1, 1, vec.shape[-1])


def mlp_embedder(p: dict[str, Array], x: Array) -> Array:
    """`MLPEmbedder` (`model.py:683-691`): bias-free 256->H->H with a SiLU between."""
    return linear(jax.nn.silu(linear(x, p["in_layer.weight"])), p["out_layer.weight"])


def qk_norm(q: Array, k: Array, query_scale: Array, key_scale: Array) -> tuple[Array, Array]:
    """`QKNorm` (`model.py:746-755`): per-head RMSNorm over `head_dim`, q and k only."""
    return rms_norm(q, query_scale), rms_norm(k, key_scale)


def heads(qkv: Array, num_heads: int) -> tuple[Array, Array, Array]:
    """`rearrange("B L (K H D) -> K B H L D", K=3)` — **K outermost**, so q/k/v in that order."""
    b, length, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    split = qkv.reshape(b, length, 3, num_heads, head_dim).transpose(2, 0, 3, 1, 4)
    return split[0], split[1], split[2]


def merge_heads(x: Array) -> Array:
    """`rearrange("b h n d -> b n (h d)")`."""
    b, h, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)


# --------------------------------------------------------------------------------------
# double-stream block
# --------------------------------------------------------------------------------------


class DoubleQKV(NamedTuple):
    """Everything `DoubleStreamBlock._prepare_qkv` returns (`model.py:569-612`)."""

    q: Array
    k: Array
    v: Array
    pe_full: Array
    num_txt: int
    img_mod1: ModTriple
    img_mod2: ModTriple
    txt_mod1: ModTriple
    txt_mod2: ModTriple


def double_prepare_qkv(
    p: dict[str, Array],
    img: Array,
    txt: Array,
    pe: Array,
    pe_ctx: Array,
    mod_img: Array,
    mod_txt: Array,
    num_heads: int,
) -> DoubleQKV:
    """Per-stream norm/qkv/QK-norm, then concatenate q/k/v on the **sequence** axis.

    `img` and `txt` stay separate tensors here — only the projected q/k/v are joint.
    """
    img_mod1, img_mod2 = split_triple(mod_img, 0), split_triple(mod_img, 3)
    txt_mod1, txt_mod2 = split_triple(mod_txt, 0), split_triple(mod_txt, 3)

    img_q, img_k, img_v = heads(linear(modulate(img, img_mod1), p["img_attn.qkv.weight"]), num_heads)
    txt_q, txt_k, txt_v = heads(linear(modulate(txt, txt_mod1), p["txt_attn.qkv.weight"]), num_heads)
    img_q, img_k = qk_norm(img_q, img_k, p["img_attn.norm.query_norm.scale"],
                           p["img_attn.norm.key_norm.scale"])
    txt_q, txt_k = qk_norm(txt_q, txt_k, p["txt_attn.norm.query_norm.scale"],
                           p["txt_attn.norm.key_norm.scale"])

    # [txt | img] on the sequence axis, and the same for the rope tables.
    q = jnp.concatenate([txt_q, img_q], axis=2)
    k = jnp.concatenate([txt_k, img_k], axis=2)
    v = jnp.concatenate([txt_v, img_v], axis=2)
    pe_full = jnp.concatenate([pe_ctx, pe], axis=2)
    q, k = apply_rope(q, k, pe_full)
    return DoubleQKV(q, k, v, pe_full, txt.shape[1], img_mod1, img_mod2, txt_mod1, txt_mod2)


def double_block(
    p: dict[str, Array],
    img: Array,
    txt: Array,
    pe: Array,
    pe_ctx: Array,
    mod_img: Array,
    mod_txt: Array,
    num_heads: int,
) -> tuple[Array, Array]:
    """`DoubleStreamBlock.forward_kv_extract` with `num_ref_tokens=0` — i.e. plain forward.

    One joint attention over `[txt, img]`, split back immediately, then per-stream residuals
    with per-stream gates and separate output projections/MLPs.
    """
    state = double_prepare_qkv(p, img, txt, pe, pe_ctx, mod_img, mod_txt, num_heads)
    attn = merge_heads(sdpa(state.q, state.k, state.v))
    txt_attn, img_attn = attn[:, : state.num_txt], attn[:, state.num_txt :]

    img = img + state.img_mod1.gate * linear(img_attn, p["img_attn.proj.weight"])
    img = img + state.img_mod2.gate * linear(
        silu_gate(linear(modulate(img, state.img_mod2), p["img_mlp.0.weight"])), p["img_mlp.2.weight"]
    )
    txt = txt + state.txt_mod1.gate * linear(txt_attn, p["txt_attn.proj.weight"])
    txt = txt + state.txt_mod2.gate * linear(
        silu_gate(linear(modulate(txt, state.txt_mod2), p["txt_mlp.0.weight"])), p["txt_mlp.2.weight"]
    )
    return img, txt


# --------------------------------------------------------------------------------------
# single-stream block
# --------------------------------------------------------------------------------------


class SingleQKV(NamedTuple):
    """`SingleStreamBlock._qkv` (`model.py:468-480`)."""

    q: Array
    k: Array
    v: Array
    mlp: Array
    gate: Array


def single_prepare_qkv(
    p: dict[str, Array], x: Array, mod: Array, num_heads: int, pe: Array
) -> SingleQKV:
    """`linear1` splits **`[qkv | mlp]`** — qkv first — then rope is applied to q/k only."""
    triple = split_triple(mod, 0)
    qkv, mlp = jnp.split(
        linear(modulate(x, triple), p["linear1.weight"]),
        [3 * x.shape[-1]],
        axis=-1,
    )
    q, k, v = heads(qkv, num_heads)
    q, k = qk_norm(q, k, p["norm.query_norm.scale"], p["norm.key_norm.scale"])
    q, k = apply_rope(q, k, pe)
    return SingleQKV(q, k, v, mlp, triple.gate)


def single_block(p: dict[str, Array], x: Array, pe: Array, mod: Array, num_heads: int) -> Array:
    """`SingleStreamBlock.forward_kv_extract` with `num_ref_tokens=0`.

    `linear2(cat([attn | silu_gate(mlp)]))`, residual scaled by the gate. `self.scale` in the
    reference is dead code and is not ported.
    """
    state = single_prepare_qkv(p, x, mod, num_heads, pe)
    attn = merge_heads(sdpa(state.q, state.k, state.v))
    out = linear(jnp.concatenate([attn, silu_gate(state.mlp)], axis=-1), p["linear2.weight"])
    return x + state.gate * out


# --------------------------------------------------------------------------------------
# final layer
# --------------------------------------------------------------------------------------


def last_layer(p: dict[str, Array], x: Array, vec: Array) -> Array:
    """`LastLayer` (`model.py:426-434`): a **2-way** `(shift, scale)` chunk, unlike the triples."""
    shift, scale = jnp.split(linear(jax.nn.silu(vec), p["adaLN_modulation.1.weight"]), 2, axis=-1)
    return linear((1 + scale[:, None]) * layer_norm(x) + shift[:, None], p["linear.weight"])


# --------------------------------------------------------------------------------------
# qwen3 decoder layer (text encoder)
# --------------------------------------------------------------------------------------


def split_heads(x: Array, num_heads: int) -> Array:
    """(B,S,H*D) -> (B,H,S,D) — one projection, not the DiT's fused qkv."""
    b, s, _ = x.shape
    return x.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)


def rotate_half(x: Array) -> Array:
    """`rotate_half`: `[-x[..., d/2:], x[..., :d/2]]` — the **half-split** convention.

    The opposite of the DiT's interleaved adjacent pairs (`rope.apply_rope`). Getting these two
    confused is silent.
    """
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def neox_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    """Qwen3's rotation: `(q * cos) + (rotate_half(q) * sin)`, all in bf16 — the reference casts
    `cos`/`sin` to the model dtype inside `Qwen3RotaryEmbedding.forward`."""
    cos, sin = cos.astype(q.dtype), sin.astype(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def qwen3_layer(
    p: dict[str, Array], x: Array, cos: Array, sin: Array, mask: Array, cfg: Qwen3_4B
) -> Array:
    """One `Qwen3DecoderLayer`: pre-norm attention (GQA + per-head q/k norm) then a gated MLP."""
    residual = x
    h = rms_norm(x, p["input_layernorm.weight"], cfg.rms_eps)
    q = split_heads(linear(h, p["self_attn.q_proj.weight"]), cfg.num_heads)
    k = split_heads(linear(h, p["self_attn.k_proj.weight"]), cfg.num_kv_heads)
    v = split_heads(linear(h, p["self_attn.v_proj.weight"]), cfg.num_kv_heads)
    q = rms_norm(q, p["self_attn.q_norm.weight"], cfg.rms_eps)
    k = rms_norm(k, p["self_attn.k_norm.weight"], cfg.rms_eps)
    q, k = neox_rope(q, k, cos, sin)
    attn = merge_heads(sdpa(q, k, v, mask))
    x = residual + linear(attn, p["self_attn.o_proj.weight"])

    residual = x
    h = rms_norm(x, p["post_attention_layernorm.weight"], cfg.rms_eps)
    h = jax.nn.silu(linear(h, p["mlp.gate_proj.weight"])) * linear(h, p["mlp.up_proj.weight"])
    return residual + linear(h, p["mlp.down_proj.weight"])


# --------------------------------------------------------------------------------------
# vae decoder blocks (diffusers AutoencoderKLFlux2)
# --------------------------------------------------------------------------------------
# every helper takes the flat torch-named dict plus a dotted prefix, e.g.
#   resnet_block(x, p["decoder"], "mid_block.resnets.0", cfg)
#   resnet_block(x, p["decoder.up_blocks"], "2.resnets.0", cfg)

# ---------------------------------------------------------------------------- layers
def conv(x: Array, p: Params, name: str, padding: str = "SAME") -> Array:
    """Conv2d + bias, NHWC in/out. 3x3 uses SAME (pad 1); the 1x1 convs use VALID (pad 0).

    Torch adds the bias inside the fp32 accumulation; `nn.conv2d` rounds to bf16 first, so this
    costs one extra bf16 rounding per conv. That and the GroupNorm round-trips are why the decoder
    measures 40-48 dB (`tests/test_vae.py`) rather than bit-exact.
    """
    return conv2d(x, p[f"{name}.weight"], padding=padding) + p[f"{name}.bias"]


def gn(x: Array, p: Params, name: str, cfg: VAE) -> Array:
    """`nn.GroupNorm(32, affine=True, eps=1e-6)` — the same eps in resnets, attention and conv_norm_out."""
    return group_norm(x, p[f"{name}.weight"], p[f"{name}.bias"], cfg.groups, cfg.norm_eps)


def resnet_block(x: Array, p: Params, prefix: str, cfg: VAE) -> Array:
    """`ResnetBlock2D` with `temb=None`: GN -> silu -> 3x3 -> GN -> silu -> 3x3, plus the skip.

    `output_scale_factor` is 1.0 and dropout is inert in eval, so both are dropped.
    """
    h = conv(jax.nn.silu(gn(x, p, f"{prefix}.norm1", cfg)), p, f"{prefix}.conv1")
    h = conv(jax.nn.silu(gn(h, p, f"{prefix}.norm2", cfg)), p, f"{prefix}.conv2")
    if f"{prefix}.conv_shortcut.weight" in p:
        x = conv(x, p, f"{prefix}.conv_shortcut", padding="VALID")
    return x + h


def attn_block(x: Array, p: Params, prefix: str, cfg: VAE) -> Array:
    """`Attention` at 1 head / `dim_head` 512, `AttnProcessor2_0`: GN -> 1x1 q,k,v -> SDPA over the
    flattened `h*w` positions -> 1x1 `to_out.0` -> residual.

    `scale_qk` gives the SDPA default `1/sqrt(512)`; `rescale_output_factor` is 1.0 so the division
    is dropped; `upcast_softmax` is dead — `AttnProcessor2_0` ignores it.
    """
    b, h, w, c = x.shape
    normed = gn(x, p, f"{prefix}.group_norm", cfg).reshape(b, h * w, c)
    q, k, v = (linear(normed, p[f"{prefix}.to_{n}.weight"]) + p[f"{prefix}.to_{n}.bias"]
               for n in ("q", "k", "v"))
    out = sdpa(q[:, None], k[:, None], v[:, None])[:, 0]
    out = linear(out, p[f"{prefix}.to_out.0.weight"]) + p[f"{prefix}.to_out.0.bias"]
    return x + out.reshape(b, h, w, c)


def upsample(x: Array, p: Params, prefix: str) -> Array:
    """`Upsample2D`: `F.interpolate(scale_factor=2, mode="nearest")` then a 3x3 conv."""
    return conv(upsample_nearest(x), p, f"{prefix}.conv")

def unpatchify(z: Array, patch: int = 2) -> Array:
    """`Flux2KleinPipeline._unpatchify_latents`: packed channel `c*patch^2 + pi*patch + pj` lands at
    pixel `(2*i + pi, 2*j + pj)` — **channel-major**, verified in `tests/test_vae.py`."""
    b, c, h, w = z.shape
    x = z.reshape(b, c // patch**2, patch, patch, h, w).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // patch**2, h * patch, w * patch)
