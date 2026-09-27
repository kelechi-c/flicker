"""Fused Pallas qk-prep for the DiT attention chain (D3c).

The chain between the q/k/v projection GEMMs and flash attention — the `heads()`
`(B,L,3,H,D) -> (3,B,H,L,D)` transpose, `qk_norm` (a per-head RMSNorm over `D`, fp32 compute ->
bf16 cast -> bf16 scale) and `apply_rope` (fp32 adjacent-pair rotation -> bf16) — costs
~114-128 ms per 4-step scan at 1024² in separate HBM round trips against a ~10-20 ms byte floor
(`out/tpu/d3b/op-profile.json`). This kernel does all three in one pass: the projection output is
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
`(1,1,L,64,2,2)` table; the two products and their sum are the same, so the result is the same
(`docs/optimization-notes.md`, 2026-09-26 D2/D3 bit-identity).

Pallas is imported inside `qk_prep` so a CPU import of this module stays free, as `nn.flash_sdpa`
does. `row_tile` returning `None` means no legal tile exists and the caller keeps the XLA chain.
"""

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
# the lane-friendly rope tables `qk_prep` consumes: `(cos, sin)`, each `(L, head_dim)` fp32
type Freqs = tuple[Array, Array]

EPS = 1e-6
# Row tiles, largest first. A program now holds only (rows, 128) tiles (~0.6 MB at 2304 rows), so
# large tiles are cheap; they must divide L. 4608 -> 2304, 4096 -> 512, 512 -> 512.
ROW_TILES = (2304, 1536, 1152, 512, 256, 128, 64)


def row_tile(length: int) -> int | None:
    """The largest `ROW_TILES` entry that divides `length`, or `None` (the caller keeps XLA)."""
    return next((size for size in ROW_TILES if length % size == 0), None)


def lane_pair(freqs: Array) -> Freqs:
    """Today's `(1,1,L,64,2,2)` fp32 rope table -> the `(L, 128)` cos/sin lanes `qk_prep` takes.

    `cos` repeats each frequency's cosine on both lanes of its pair and `sin` is `-sin` on even
    lanes, `+sin` on odd ones, so `cos*x + sin*pair_swap(x)` is exactly the packed
    `[[cos,-sin],[sin,cos]]` rotation — the same two products, the same sum.
    """
    cos, sin = freqs[..., 0, 0][0, 0], freqs[..., 1, 0][0, 0]
    lanes = jnp.stack([-sin, sin], axis=-1).reshape(*sin.shape[:-1], 2 * sin.shape[-1])
    return jnp.repeat(cos, 2, axis=-1), lanes


def _body(q_in, k_in, v_in, cos, sin, q_scale, k_scale, q_out, k_out, v_out):
    """One `(rows, 128)` tile of one head: RMSNorm + rope on q and k, copy v. All 2-D."""
    from jax.experimental.pallas import tpu as pltpu

    c, s = cos[...], sin[...]
    width = c.shape[-1]
    even = lax.broadcasted_iota(jnp.int32, c.shape, 1) % 2 == 0

    def prep(value, scale):
        wide = value.astype(jnp.float32)
        rms = jnp.mean(wide * wide, axis=-1, keepdims=True)
        normed = (wide * lax.rsqrt(rms + EPS)).astype(value.dtype) * scale
        wide = normed.astype(jnp.float32)
        # even lane 2d takes x[2d+1] (roll by -1), odd lane takes x[2d] (roll by +1)
        swap = jnp.where(even, pltpu.roll(wide, width - 1, 1), pltpu.roll(wide, 1, 1))
        return (c * wide + s * swap).astype(value.dtype)

    q_out[0, 0] = prep(q_in[0], q_scale[...])
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
    `modules.heads` splits it; `freqs` is this stream's `lane_pair` table. Batch 1 only (the DiT's
    serving shape). `rows` overrides the tile (the bench sweeps it); `interpret` runs on CPU.
    """
    import jax.experimental.pallas as pl

    batch, length, width = qkv.shape
    head_dim = width // (3 * num_heads)
    tile_rows = rows or row_tile(length)
    if batch != 1 or tile_rows is None or length % tile_rows:
        raise ValueError(f"qk_prep needs batch 1 and a row tile dividing L={length} (rows={tile_rows})")
    cos, sin = freqs

    def column(offset: int) -> "pl.BlockSpec":
        """Head `h`'s 128-wide slice of the q (offset 0), k (H) or v (2H) columns."""
        return pl.BlockSpec((1, tile_rows, head_dim), lambda i, h: (0, i, offset + h))

    table = pl.BlockSpec((tile_rows, head_dim), lambda i, h: (i, 0))
    scale = pl.BlockSpec((1, head_dim), lambda i, h: (0, 0))
    out = pl.BlockSpec((1, 1, tile_rows, head_dim), lambda i, h: (0, h, i, 0))
    kernel = pl.pallas_call(
        _body,
        out_shape=[jax.ShapeDtypeStruct((batch, num_heads, length, head_dim), qkv.dtype)] * 3,
        grid=(length // tile_rows, num_heads),
        in_specs=[column(0), column(num_heads), column(2 * num_heads), table, table, scale, scale],
        out_specs=[out] * 3,
        interpret=interpret,
        name="qk_prep",
    )
    return tuple(kernel(qkv, qkv, qkv, cos, sin, q_scale.reshape(1, -1), k_scale.reshape(1, -1)))
