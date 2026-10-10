import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


class Freqs(NamedTuple):

    cos: Array
    sin: Array
    q_gain: float = 1.0


EPS = 1e-6
ROW_TILES = (2304, 1536, 1152, 512, 256, 128, 64)


def row_tile(length: int) -> int | None:
    return next((size for size in ROW_TILES if length % size == 0), None)


# rope table (1,1,L,64,2,2) -> (L,128) cos/sin lanes, so rotation is cos*x + sin*pair_swap(x)
def lane_pair(freqs: Array, q_gain: float = 1.0) -> Freqs:
    cos, sin = freqs[..., 0, 0][0, 0], freqs[..., 1, 0][0, 0]
    lanes = jnp.stack([-sin, sin], axis=-1).reshape(*sin.shape[:-1], 2 * sin.shape[-1])
    return Freqs(jnp.repeat(cos, 2, axis=-1), lanes, q_gain)


# one (rows, 128) tile of one head: rmsnorm + rope on q and k, plain copy of v
def _body(q_in, k_in, v_in, cos, sin, q_scale, k_scale, q_out, k_out, v_out, *, q_gain=1.0):
    from jax.experimental.pallas import tpu as pltpu

    c, s = cos[...], sin[...]
    width = c.shape[-1]
    even = lax.broadcasted_iota(jnp.int32, c.shape, 1) % 2 == 0

    def prep(value, scale, gain=1.0):
        wide = value.astype(jnp.float32)
        rms = jnp.mean(wide * wide, axis=-1, keepdims=True)
        normed = (wide * lax.rsqrt(rms + EPS)).astype(value.dtype) * scale
        wide = normed.astype(jnp.float32)
        swap = jnp.where(even, pltpu.roll(wide, width - 1, 1), pltpu.roll(wide, 1, 1))
        rotated = c * wide + s * swap
        return (rotated if gain == 1.0 else gain * rotated).astype(value.dtype)

    q_out[0, 0] = prep(q_in[0], q_scale[...], q_gain)
    k_out[0, 0] = prep(k_in[0], k_scale[...])
    v_out[0, 0] = v_in[0]


# fused head transpose + qk rmsnorm + rope in one pass; the output index map does the transpose
def qk_prep(
    qkv: Array,
    q_scale: Array,
    k_scale: Array,
    freqs: Freqs,
    num_heads: int,
    *,
    rows: int | None = None,
    interpret: bool = False,
) -> tuple[Array, Array, Array]:
    import jax.experimental.pallas as pl

    batch, length, width = qkv.shape
    head_dim = width // (3 * num_heads)
    tile_rows = rows or row_tile(length)
    if tile_rows is None or length % tile_rows:
        raise ValueError(f"qk_prep needs a row tile dividing L={length} (rows={tile_rows})")
    cos, sin, q_gain = freqs

    def column(offset: int) -> "pl.BlockSpec":
        return pl.BlockSpec((1, tile_rows, head_dim), lambda b, i, h: (b, i, offset + h))

    table = pl.BlockSpec((tile_rows, head_dim), lambda b, i, h: (i, 0))
    scale = pl.BlockSpec((1, head_dim), lambda b, i, h: (0, 0))
    out = pl.BlockSpec((1, 1, tile_rows, head_dim), lambda b, i, h: (b, h, i, 0))
    kernel = pl.pallas_call(
        functools.partial(_body, q_gain=q_gain),
        out_shape=[jax.ShapeDtypeStruct((batch, num_heads, length, head_dim), qkv.dtype)] * 3,
        grid=(batch, length // tile_rows, num_heads),
        in_specs=[column(0), column(num_heads), column(2 * num_heads), table, table, scale, scale],
        out_specs=[out] * 3,
        interpret=interpret,
        name="qk_prep",
    )
    return tuple(kernel(qkv, qkv, qkv, cos, sin, q_scale.reshape(1, -1), k_scale.reshape(1, -1)))


SCORE_TILES = (2304, 1536, 1152, 512, 256, 128)
EXP_LIMIT = 80.0


# upper bound on |q.k|/sqrt(d) implied by the qk-norm scales; must stay under exp's fp32 overflow point
def logit_bound(q_scale: Array, k_scale: Array) -> Array:
    width = q_scale.shape[-1]
    return math.sqrt(width) * jnp.max(jnp.abs(q_scale), -1) * jnp.max(jnp.abs(k_scale), -1)


# flash attention without the running max (logits are bounded), q pre-scaled by 1/sqrt(d)
def attend(q: Array, k: Array, v: Array, *, block: int | None = None, interpret: bool = False) -> Array:
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    batch, heads, length, dim = q.shape
    block = block or next(size for size in SCORE_TILES if length % size == 0)
    steps = length // block

    def body(q_ref, k_ref, v_ref, o_ref, l_ref, acc_ref):
        step = pl.program_id(3)

        @pl.when(step == 0)
        def _():
            l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        tile_v = v_ref[0, 0]
        s = lax.dot_general(q_ref[0, 0], k_ref[0, 0], (((1,), (1,)), ((), ())),
                            preferred_element_type=jnp.float32)
        p = jnp.exp(s).astype(tile_v.dtype)
        l_ref[...] += jnp.sum(p.astype(jnp.float32), axis=1)[:, None]
        acc_ref[...] += lax.dot(p, tile_v, preferred_element_type=jnp.float32)

        @pl.when(step == steps - 1)
        def _():
            o_ref[0, 0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)

    rows = pl.BlockSpec((1, 1, block, dim), lambda b, h, i, j: (b, h, i, 0))
    cols = pl.BlockSpec((1, 1, block, dim), lambda b, h, i, j: (b, h, j, 0))
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=(batch, heads, steps, steps),
            in_specs=[rows, cols, cols], out_specs=rows,
            scratch_shapes=[pltpu.VMEM((block, dim), jnp.float32)] * 2),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
        interpret=interpret,
        name="attend",
    )(q, k, v)
