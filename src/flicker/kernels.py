"""Fused Pallas qk-prep for the DiT attention chain.

The chain between the q/k/v projection GEMMs and flash attention — the `heads()`
`(B,L,3,H,D) -> (3,B,H,L,D)` transpose, `qk_norm` (a per-head RMSNorm over `D`, fp32 compute ->
bf16 cast -> bf16 scale) and `apply_rope` (fp32 adjacent-pair rotation -> bf16) — costs
~114-128 ms per 4-step scan at 1024² in separate HBM round trips against a ~10-20 ms byte floor. This kernel does all three in one pass: the projection output is
read once, q/k/v are written once, and the head transpose is folded into the output index map.

Mosaic shape rules (the first version failed to lower: strided lane slices became a gather, and
it reshaped/transposed tiles in registers). This version only ever touches 2-D `(rows, 128)` tiles:

- **The grid is (row tile, head).** Three `BlockSpec`s pick head `h`'s 128-wide q, k and v column
  slices straight out of the `(B, L, 3*H*D)` projection output, and the output index map
  `(0, h, i, 0)` drops each tile into its `(B, H, L, D)` slot — the DMA does the head transpose.
- **The pair swap is `pltpu.roll`** (a native lane rotate) plus a parity select.
- **v is a pure layout copy** in the same pass.

The arithmetic is the shipped chain's, cast point for cast point: `fp32 mean of squares + eps ->
rsqrt -> multiply -> cast bf16 -> times the bf16 QKNorm scale`, then `fp32 -> cos*x + sin*pair_swap
-> cast bf16`. `lane_pair` derives the `(L, 128)` lanes this kernel wants from today's
`(1,1,L,64,2,2)` table; the two products and their sum are the same, so the result is the same.

Pallas is imported inside `qk_prep` so a CPU import of this module stays free, as `nn.flash_sdpa`
does. `row_tile` returning `None` means no legal tile exists and the caller keeps the XLA chain.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


class Freqs(NamedTuple):
    """The lane-friendly rope tables `qk_prep` consumes, each `(L, head_dim)` fp32.

    `q_gain` multiplies q (not k) in fp32 just before its one bf16 rounding. `attend` wants
    `1/sqrt(D)` there, so the attention scale costs no pass of its own and no second rounding.
    """

    cos: Array
    sin: Array
    q_gain: float = 1.0


EPS = 1e-6
# Row tiles, largest first. A program now holds only (rows, 128) tiles (~0.6 MB at 2304 rows), so
# large tiles are cheap; they must divide L. 4608 -> 2304, 4096 -> 512, 512 -> 512.
ROW_TILES = (2304, 1536, 1152, 512, 256, 128, 64)


def row_tile(length: int) -> int | None:
    """The largest `ROW_TILES` entry that divides `length`, or `None` (the caller keeps XLA)."""
    return next((size for size in ROW_TILES if length % size == 0), None)


def lane_pair(freqs: Array, q_gain: float = 1.0) -> Freqs:
    """Today's `(1,1,L,64,2,2)` fp32 rope table -> the `(L, 128)` cos/sin lanes `qk_prep` takes.

    `cos` repeats each frequency's cosine on both lanes of its pair and `sin` is `-sin` on even
    lanes, `+sin` on odd ones, so `cos*x + sin*pair_swap(x)` is exactly the packed
    `[[cos,-sin],[sin,cos]]` rotation — the same two products, the same sum.
    """
    cos, sin = freqs[..., 0, 0][0, 0], freqs[..., 1, 0][0, 0]
    lanes = jnp.stack([-sin, sin], axis=-1).reshape(*sin.shape[:-1], 2 * sin.shape[-1])
    return Freqs(jnp.repeat(cos, 2, axis=-1), lanes, q_gain)


def _body(q_in, k_in, v_in, cos, sin, q_scale, k_scale, q_out, k_out, v_out, *, q_gain=1.0):
    """One `(rows, 128)` tile of one head: RMSNorm + rope on q and k, copy v. All 2-D."""
    from jax.experimental.pallas import tpu as pltpu

    c, s = cos[...], sin[...]
    width = c.shape[-1]
    even = lax.broadcasted_iota(jnp.int32, c.shape, 1) % 2 == 0

    def prep(value, scale, gain=1.0):
        wide = value.astype(jnp.float32)
        rms = jnp.mean(wide * wide, axis=-1, keepdims=True)
        normed = (wide * lax.rsqrt(rms + EPS)).astype(value.dtype) * scale
        wide = normed.astype(jnp.float32)
        # even lane 2d takes x[2d+1] (roll by -1), odd lane takes x[2d] (roll by +1)
        swap = jnp.where(even, pltpu.roll(wide, width - 1, 1), pltpu.roll(wide, 1, 1))
        rotated = c * wide + s * swap
        return (rotated if gain == 1.0 else gain * rotated).astype(value.dtype)

    q_out[0, 0] = prep(q_in[0], q_scale[...], q_gain)
    k_out[0, 0] = prep(k_in[0], k_scale[...])
    v_out[0, 0] = v_in[0]


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
    """One stream's `(B, L, 3*H*D)` projection output -> `q, k, v` `(B, H, L, D)` bf16.

    `qkv` is exactly what `nn.linear` produces, `[q | k | v]` on the minor axis in the order
    `modules.heads` splits it; `freqs` is this stream's `lane_pair` table (shared across the batch).
    `rows` overrides the tile (the bench sweeps it); `interpret` runs on CPU.
    """
    import jax.experimental.pallas as pl

    batch, length, width = qkv.shape
    head_dim = width // (3 * num_heads)
    tile_rows = rows or row_tile(length)
    if tile_rows is None or length % tile_rows:
        raise ValueError(f"qk_prep needs a row tile dividing L={length} (rows={tile_rows})")
    cos, sin, q_gain = freqs

    def column(offset: int) -> "pl.BlockSpec":
        """Head `h`'s 128-wide slice of the q (offset 0), k (H) or v (2H) columns."""
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


# Score tiles for `attend`, largest first; q and k use the same one. 2304 x 2304 fp32 scores are
# 21 MB, under the compiler's default 32 MB scoped-vmem limit per kernel (a budget, not the chip's
# vmem; `vmem_limit_bytes` raises it), and the v6e sweep (`tools/bench_flash.py`) ranks it first
# at L=4608.
# every entry is a multiple of 128, the lane width the kernel's tiles need.
SCORE_TILES = (2304, 1536, 1152, 512, 256, 128)
# fp32 `exp` overflows at 88.7; `attend` is only legal below this.
EXP_LIMIT = 80.0


def logit_bound(q_scale: Array, k_scale: Array) -> Array:
    """The largest `|q . k| / sqrt(D)` QK-norm allows, per block.

    RMSNorm leaves every q and k row with mean square 1, the per-lane scale stretches it by at most
    `max|scale|`, and rope is a rotation, so `|q| <= sqrt(D) max|s_q|` (same for k) and
    Cauchy-Schwarz gives `|q . k| / sqrt(D) <= sqrt(D) max|s_q| max|s_k|`.
    """
    width = q_scale.shape[-1]
    return math.sqrt(width) * jnp.max(jnp.abs(q_scale), -1) * jnp.max(jnp.abs(k_scale), -1)


def attend(q: Array, k: Array, v: Array, *, block: int | None = None, interpret: bool = False) -> Array:
    """`softmax(q k^T) v` over `(B, H, L, D)`, streamed in tiles, **without a running max**.

    `q` must already carry the `1/sqrt(D)` attention scale (`Freqs.q_gain`): rescaling the score
    tile inside the kernel costs 0.1 ms per call on v6e.

    Flash attention carries a running row max so `exp` cannot overflow: per k tile that is a max
    pass and a subtract over the whole score tile, plus a rescale of everything accumulated so far.
    Here the logits are bounded by construction (`logit_bound` < `EXP_LIMIT`), so `exp(s)` is safe
    as it stands and the softmax is two running sums and one divide:

        p = bf16(exp(s))      acc += p @ v      l += sum(p)      out = acc / l

    `l` sums the **rounded** weights, the ones the matmul sees. Summing the fp32 `exp` instead
    leaves each row scaled by its dominant weight's rounding error (up to 0.4 %), which the
    running-max form never shows because its largest weight is `exp(0) = 1`, exact in bf16.

    The grid is (batch, head, q tile, k tile); the k axis is the accumulation, held in vmem scratch
    and written out on its last step. Measured on v6e-1 at L=4608: 0.69 ms per call against the
    library flash kernel's 1.02 ms.
    """
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
