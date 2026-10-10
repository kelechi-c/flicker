import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Padding = str | tuple[tuple[int, int], tuple[int, int]]
Attention = Callable[[Array, Array, Array], Array]


def linear(x: Array, w: Array) -> Array:
    return x @ w


def layer_norm(x: Array, eps: float = 1e-6) -> Array:
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * lax.rsqrt(var + eps)).astype(x.dtype)


def rms_norm(x: Array, scale: Array, eps: float = 1e-6) -> Array:
    xf = x.astype(jnp.float32)
    normed = xf * lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * scale


def silu_gate(x: Array) -> Array:
    gate, value = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(gate) * value


def sdpa(q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
    qf, kf, vf = (a.astype(jnp.float32) for a in (q, k, v))
    if qf.shape[1] != kf.shape[1]:
        repeat = qf.shape[1] // kf.shape[1]
        kf, vf = jnp.repeat(kf, repeat, axis=1), jnp.repeat(vf, repeat, axis=1)
    logits = jnp.einsum("bhld,bhsd->bhls", qf, kf) * (1.0 / math.sqrt(q.shape[-1]))
    if mask is not None:
        logits = jnp.where(mask, logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhls,bhsd->bhld", weights, vf).astype(v.dtype)


def flash_sdpa(q: Array, k: Array, v: Array) -> Array:
    import jax.experimental.pallas.ops.tpu.flash_attention as flash

    length = q.shape[2]
    if length % 128:
        return sdpa(q, k, v)
    block_q = tile(length, (2304, 1536, 1152, 512, 256, 128))
    block_k = tile(length, (1536, 1152, 512, 256, 128))
    return flash.flash_attention(
        q, k, v, causal=False, sm_scale=1.0 / math.sqrt(q.shape[-1]),
        block_sizes=flash.BlockSizes(block_q=block_q, block_k_major=block_k, block_k=block_k,
                                     block_b=1))


def tile(length: int, sizes: tuple[int, ...]) -> int:
    return next(size for size in sizes if length % size == 0)


def conv2d(x: Array, w: Array, stride: int = 1, padding: Padding = "SAME", dilation: int = 1) -> Array:
    out = lax.conv_general_dilated(
        x,
        w,
        (stride, stride),
        padding.upper() if isinstance(padding, str) else padding,
        lhs_dilation=(dilation, dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    )
    return out.astype(x.dtype)


def group_norm(x: Array, weight: Array, bias: Array, groups: int = 32, eps: float = 1e-6) -> Array:
    n, h, w, c = x.shape
    xf = x.astype(jnp.float32).reshape(n, h * w, groups, c // groups)
    mean = jnp.mean(xf, axis=(1, 3), keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=(1, 3), keepdims=True)
    normed = ((xf - mean) * lax.rsqrt(var + eps)).reshape(n, h, w, c)
    return (normed * weight.astype(jnp.float32) + bias.astype(jnp.float32)).astype(x.dtype)


def upsample_nearest(x: Array, factor: int = 2) -> Array:
    n, h, w, c = x.shape
    return jnp.broadcast_to(x[:, :, None, :, None, :], (n, h, factor, w, factor, c)).reshape(
        n, h * factor, w * factor, c
    )


def timestep_embedding(t: Array, dim: int = 256, max_period: float = 10000.0,
                       time_factor: float = 1000.0) -> Array:
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = (t * time_factor).astype(jnp.float32)[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding.astype(t.dtype)
