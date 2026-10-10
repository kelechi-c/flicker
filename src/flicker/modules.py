from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import VAE, Qwen3_4B
from .kernels import Freqs, qk_prep, row_tile
from .nn import (
    Attention,
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

Mod = Array


class ModTriple(NamedTuple):

    shift: Array
    scale: Array
    gate: Array


def split_triple(mod: Array, index: int) -> ModTriple:
    return ModTriple(mod[:, index], mod[:, index + 1], mod[:, index + 2])


def modulate(x: Array, mod: ModTriple, norm: bool = True) -> Array:
    base = layer_norm(x) if norm else x
    return (1 + mod.scale) * base + mod.shift


def modulation(p: dict[str, Array], vec: Array, double: bool) -> Array:
    out = linear(jax.nn.silu(vec), p["lin.weight"])
    return out[:, None, :].reshape(out.shape[0], -1, 1, vec.shape[-1])


def mlp_embedder(p: dict[str, Array], x: Array) -> Array:
    return linear(jax.nn.silu(linear(x, p["in_layer.weight"])), p["out_layer.weight"])


def qk_norm(q: Array, k: Array, query_scale: Array, key_scale: Array) -> tuple[Array, Array]:
    return rms_norm(q, query_scale), rms_norm(k, key_scale)


def heads(qkv: Array, num_heads: int) -> tuple[Array, Array, Array]:
    b, length, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    split = qkv.reshape(b, length, 3, num_heads, head_dim).transpose(2, 0, 3, 1, 4)
    return split[0], split[1], split[2]


def merge_heads(x: Array) -> Array:
    b, h, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)


def qkv_prep(
    qkv: Array,
    q_scale: Array,
    k_scale: Array,
    pe: Array,
    freqs: Freqs | None,
    num_heads: int,
) -> tuple[Array, Array, Array]:
    if freqs is not None and row_tile(qkv.shape[1]) is not None:
        return qk_prep(qkv, q_scale, k_scale, freqs, num_heads)
    q, k, v = heads(qkv, num_heads)
    q, k = apply_rope(*qk_norm(q, k, q_scale, k_scale), pe)
    return q, k, v


class DoubleQKV(NamedTuple):

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
    freqs: Freqs | None = None,
) -> DoubleQKV:
    img_mod1, img_mod2 = split_triple(mod_img, 0), split_triple(mod_img, 3)
    txt_mod1, txt_mod2 = split_triple(mod_txt, 0), split_triple(mod_txt, 3)
    num_txt = txt.shape[1]

    if freqs is None:
        img_q, img_k, img_v = heads(linear(modulate(img, img_mod1), p["img_attn.qkv.weight"]),
                                    num_heads)
        txt_q, txt_k, txt_v = heads(linear(modulate(txt, txt_mod1), p["txt_attn.qkv.weight"]),
                                    num_heads)
        img_q, img_k = qk_norm(img_q, img_k, p["img_attn.norm.query_norm.scale"],
                               p["img_attn.norm.key_norm.scale"])
        txt_q, txt_k = qk_norm(txt_q, txt_k, p["txt_attn.norm.query_norm.scale"],
                               p["txt_attn.norm.key_norm.scale"])
        q = jnp.concatenate([txt_q, img_q], axis=2)
        k = jnp.concatenate([txt_k, img_k], axis=2)
        v = jnp.concatenate([txt_v, img_v], axis=2)
        pe_full = jnp.concatenate([pe_ctx, pe], axis=2)
        q, k = apply_rope(q, k, pe_full)
        return DoubleQKV(q, k, v, pe_full, num_txt, img_mod1, img_mod2, txt_mod1, txt_mod2)

    txt_q, txt_k, txt_v = qkv_prep(
        linear(modulate(txt, txt_mod1), p["txt_attn.qkv.weight"]),
        p["txt_attn.norm.query_norm.scale"], p["txt_attn.norm.key_norm.scale"], pe_ctx,
        Freqs(freqs.cos[:num_txt], freqs.sin[:num_txt], freqs.q_gain), num_heads)
    img_q, img_k, img_v = qkv_prep(
        linear(modulate(img, img_mod1), p["img_attn.qkv.weight"]),
        p["img_attn.norm.query_norm.scale"], p["img_attn.norm.key_norm.scale"], pe,
        Freqs(freqs.cos[num_txt:], freqs.sin[num_txt:], freqs.q_gain), num_heads)

    q = jnp.concatenate([txt_q, img_q], axis=2)
    k = jnp.concatenate([txt_k, img_k], axis=2)
    v = jnp.concatenate([txt_v, img_v], axis=2)
    pe_full = jnp.concatenate([pe_ctx, pe], axis=2)
    return DoubleQKV(q, k, v, pe_full, num_txt, img_mod1, img_mod2, txt_mod1, txt_mod2)


# joint img/txt attention with separate per-stream weights, modulation and mlps
def double_block(
    p: dict[str, Array],
    img: Array,
    txt: Array,
    pe: Array,
    pe_ctx: Array,
    mod_img: Array,
    mod_txt: Array,
    num_heads: int,
    attend: Attention = sdpa,
    freqs: Freqs | None = None,
) -> tuple[Array, Array]:
    state = double_prepare_qkv(p, img, txt, pe, pe_ctx, mod_img, mod_txt, num_heads, freqs)
    attn = merge_heads(attend(state.q, state.k, state.v))
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


class SingleQKV(NamedTuple):

    q: Array
    k: Array
    v: Array
    mlp: Array
    gate: Array


def single_prepare_qkv(
    p: dict[str, Array], x: Array, mod: Array, num_heads: int, pe: Array,
    freqs: Freqs | None = None,
) -> SingleQKV:
    triple = split_triple(mod, 0)
    qkv, mlp = jnp.split(
        linear(modulate(x, triple), p["linear1.weight"]),
        [3 * x.shape[-1]],
        axis=-1,
    )
    q, k, v = qkv_prep(qkv, p["norm.query_norm.scale"], p["norm.key_norm.scale"], pe, freqs, num_heads)
    return SingleQKV(q, k, v, mlp, triple.gate)


# fused qkv+mlp projection on the concatenated stream, attention, fused output projection
def single_block(p: dict[str, Array], x: Array, pe: Array, mod: Array, num_heads: int,
                 attend: Attention = sdpa, freqs: Freqs | None = None) -> Array:
    state = single_prepare_qkv(p, x, mod, num_heads, pe, freqs)
    attn = merge_heads(attend(state.q, state.k, state.v))
    out = linear(jnp.concatenate([attn, silu_gate(state.mlp)], axis=-1), p["linear2.weight"])
    return x + state.gate * out


def last_layer(p: dict[str, Array], x: Array, vec: Array) -> Array:
    shift, scale = jnp.split(linear(jax.nn.silu(vec), p["adaLN_modulation.1.weight"]), 2, axis=-1)
    return linear((1 + scale[:, None]) * layer_norm(x) + shift[:, None], p["linear.weight"])


def split_heads(x: Array, num_heads: int) -> Array:
    b, s, _ = x.shape
    return x.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)


def rotate_half(x: Array) -> Array:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def neox_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    cos, sin = cos.astype(q.dtype), sin.astype(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# qwen3 decoder layer: gqa attention with qk-norm and neox rope, swiglu mlp
def qwen3_layer(
    p: dict[str, Array], x: Array, cos: Array, sin: Array, mask: Array, cfg: Qwen3_4B
) -> Array:
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


def conv(x: Array, p: Params, name: str, padding: str = "SAME") -> Array:
    return conv2d(x, p[f"{name}.weight"], padding=padding) + p[f"{name}.bias"]


def gn(x: Array, p: Params, name: str, cfg: VAE) -> Array:
    return group_norm(x, p[f"{name}.weight"], p[f"{name}.bias"], cfg.groups, cfg.norm_eps)


def resnet_block(x: Array, p: Params, prefix: str, cfg: VAE) -> Array:
    h = conv(jax.nn.silu(gn(x, p, f"{prefix}.norm1", cfg)), p, f"{prefix}.conv1")
    h = conv(jax.nn.silu(gn(h, p, f"{prefix}.norm2", cfg)), p, f"{prefix}.conv2")
    if f"{prefix}.conv_shortcut.weight" in p:
        x = conv(x, p, f"{prefix}.conv_shortcut", padding="VALID")
    return x + h


def attn_block(x: Array, p: Params, prefix: str, cfg: VAE) -> Array:
    b, h, w, c = x.shape
    normed = gn(x, p, f"{prefix}.group_norm", cfg).reshape(b, h * w, c)
    q, k, v = (linear(normed, p[f"{prefix}.to_{n}.weight"]) + p[f"{prefix}.to_{n}.bias"]
               for n in ("q", "k", "v"))
    out = sdpa(q[:, None], k[:, None], v[:, None])[:, 0]
    out = linear(out, p[f"{prefix}.to_out.0.weight"]) + p[f"{prefix}.to_out.0.bias"]
    return x + out.reshape(b, h, w, c)


def box_kernel(w: Array) -> Array:
    wide = w.astype(jnp.float32)
    box = jnp.zeros((4, 4, *w.shape[2:]), jnp.float32)
    return box.at[:3, :3].add(wide).at[1:, :3].add(wide).at[:3, 1:].add(wide).at[1:, 1:].add(wide).astype(w.dtype)


# dilated: nearest-2x + 3x3 conv as one 4x4 conv on the zero-inserted input; nearest is the reference form
def upsample(x: Array, p: Params, prefix: str, cfg: VAE) -> Array:
    if cfg.upsample == "dilated":
        w = box_kernel(p[f"{prefix}.conv.weight"])
        return conv2d(x, w, padding=((2, 2), (2, 2)), dilation=2) + p[f"{prefix}.conv.bias"]
    return conv(upsample_nearest(x), p, f"{prefix}.conv")

def unpatchify(z: Array, patch: int = 2) -> Array:
    b, c, h, w = z.shape
    x = z.reshape(b, c // patch**2, patch, patch, h, w).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // patch**2, h * patch, w * patch)
