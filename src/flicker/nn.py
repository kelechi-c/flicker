"""The entire "framework": ~10 pure functions, no flax/linen/nnx.

Every one of these is a place where the dtype had to be chosen deliberately, which is why they
are hand-written rather than imported (`docs/port-plan.md` §1). Param leaves are torch-named:
`linear` takes `(in, out)` (transposed at load), `conv2d` takes `(h, w, in, out)` (OIHW -> HWIO).
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Padding = str | tuple[tuple[int, int], tuple[int, int]]
Attention = Callable[[Array, Array, Array], Array]

# int8 is symmetric, so 127 (not 128) is the positive ceiling and -127 the negative one: values
# live in [-127, 127], which keeps the round trip `dequant(quant(w))` unbiased in sign.
INT8_MAX = 127.0
# a zero row (or a zero-weight input channel) would divide by zero; the floor is far below any
# real activation's absmax, so it only ever fires on an all-zero row, where the quantized value is
# 0 regardless of the scale.
SCALE_FLOOR = 1e-6


class Int8Weight(NamedTuple):
    """A per-output-channel int8 weight: `w ~= values * scale`, `scale` fp32 `(..., out)`.

    `smooth`, when it is an array, is the fp32 per-input-channel divisor the weight was folded
    against (SmoothQuant, `quantize_weight_smooth`): `values` quantize `diag(s) w`, and
    `linear_int8` divides the activation by `s` before quantizing it, so
    `(x / s) @ (s w) == x @ w` up to the two quantizations. `None` is the un-smoothed path.
    """

    values: Array  # (..., in, out) int8
    scale: Array  # (..., out) fp32
    smooth: Array | None = None  # (..., in) fp32, or None for no smoothing


def smooth_scales(x_absmax: Array, w_absmax: Array, alpha: float) -> Array:
    """SmoothQuant's per-input-channel scale `s = max|X|^a / max|W|^(1-a)`, floored away from 0.

    Both arguments are per **input** channel: `x_absmax` over the calibration set's every token and
    every step, `w_absmax` the weight's own max over its output axis. A channel whose activations
    are loud and whose weights are quiet gets `s > 1`, which shrinks the activation (where an
    outlier costs the whole row its int8 range) and grows the weight (where a per-output-channel
    scale absorbs it for free). Pyramid `alpha` splits which side absorbs the outlier.
    """
    return (jnp.maximum(x_absmax, SCALE_FLOOR) ** alpha
            / jnp.maximum(w_absmax, SCALE_FLOOR) ** (1.0 - alpha))


def quantize_weight(w: Array) -> Int8Weight:
    """Symmetric per-output-channel int8 quantization of a linear weight, in one device pass.

    `scale = absmax over (..., in) / 127`; the rounding is half-to-even (as everywhere in JAX) and
    the clip cannot fire on a weight quantized against its own absmax — it is there so a leaf
    quantized against some *other* scale cannot wrap into the opposite sign. Pure; `w` is untouched.
    """
    wide = w.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(wide), axis=-2), SCALE_FLOOR) / INT8_MAX
    return Int8Weight(jnp.clip(jnp.round(wide / scale[..., None, :]), -INT8_MAX, INT8_MAX)
                      .astype(jnp.int8), scale)


def quantize_weight_smooth(w: Array, s: Array) -> Int8Weight:
    """Fold a per-input-channel scale into the weight (`w' = diag(s) w`), then quantize it.

    `s` broadcasts on the weight's input axis (`(..., in, out)` against `(..., in)`). The fold is
    fp32 and happens **once**, at load: the returned leaf stores the quantized `w'` plus `s`, so
    the runtime GEMM is one fused divide and the same W8A8 path the plain mode uses.
    """
    folded = quantize_weight(w.astype(jnp.float32) * s[..., None])
    return Int8Weight(folded.values, folded.scale, s)


def quantize_rows(x: Array) -> tuple[Array, Array]:
    """Dynamic per-token (row) absmax quantization of an activation -> (int8, fp32 scale `(..., 1)`).

    Per-row, not per-tensor: one outlier token must not cost every other token its precision.
    """
    wide = x.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(wide), axis=-1, keepdims=True), SCALE_FLOOR) / INT8_MAX
    return jnp.clip(jnp.round(wide / scale), -INT8_MAX, INT8_MAX).astype(jnp.int8), scale


def linear_int8(x: Array, w: Int8Weight) -> Array:
    """W8A8: dynamic per-token activation quant, int8xint8 -> int32 MXU GEMM, per-channel dequant.

    The int32 accumulation is exact (int8 products are <= 2^14 and a K=12288 reduction stays far
    inside int32), so the quantized path's entire error is the two quantizations — one rounding per
    activation element and one per weight element. Dequant is fp32 and casts once, to the
    activation's dtype, so the result is a drop-in `x @ w`. When the leaf carries a SmoothQuant
    `smooth` vector the activation is divided by it first (an elementwise op XLA fuses into the
    quantize), which is the whole runtime cost of the outlier handling.
    """
    rows, row_scale = quantize_rows(x if w.smooth is None else x / w.smooth)
    dims = (((rows.ndim - 1,), (0,)), ((), ()))
    acc = lax.dot_general(rows, w.values, dims, preferred_element_type=jnp.int32)
    return (acc.astype(jnp.float32) * row_scale * w.scale).astype(x.dtype)


def linear(x: Array, w: Array | Int8Weight) -> Array:
    """`nn.Linear` with bias=False. Torch accumulates bf16 GEMMs in fp32, as do XLA's backends.

    An `Int8Weight` leaf (built by `loader.fold_smooth`) dispatches to the W8A8 path; every other
    leaf is the plain bf16 `x @ w`. The dispatch is on the *weight's type*, so it is the loader's
    choice of leaves — a static, pre-compile property — that decides which GEMMs are quantized.
    """
    if isinstance(w, Int8Weight):
        return linear_int8(x, w)
    return x @ w


def layer_norm(x: Array, eps: float = 1e-6) -> Array:
    """`nn.LayerNorm(elementwise_affine=False)`: fp32 stats, biased variance, cast back.

    `jnp.mean` is dtype-preserving, so the upcast is mandatory to match torch.
    """
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * lax.rsqrt(var + eps)).astype(x.dtype)


def rms_norm(x: Array, scale: Array, eps: float = 1e-6) -> Array:
    """BFL's `RMSNorm` (`model.py:734-743`): fp32 compute -> **cast to bf16** -> times bf16 scale.

    Two roundings, not one. This is the load-bearing cast point (project.md §8.2 item 3).
    """
    xf = x.astype(jnp.float32)
    normed = xf * lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * scale


def silu_gate(x: Array) -> Array:
    """`SiLUActivation` (`model.py:390-397`): chunk the **last** axis, `silu(first) * second`."""
    gate, value = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(gate) * value


def sdpa(q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
    """Scaled dot-product attention over BSND, fp32 logits/softmax, output in v's dtype.

    `mask` is a boolean `(B,1,L,S)` keep-mask (text-encoder padding); the DiT passes none.
    GQA (q heads a multiple of kv heads) is handled by repeating k/v. The scale is the SDPA
    default `1/sqrt(D)` — `SingleStreamBlock.scale` is dead code (`model.py:449`).
    """
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
    """DiT attention on the Pallas TPU flash kernel: the LxL scores are never materialised.

    `sdpa` writes a `(24, 4608, 4608)` fp32 score buffer per block (2.04 GB, ~800 GB of traffic
    per image) — D0 measured that as the DiT's dominant cost. This kernel streams K/V in tiles
    with an online softmax and emits only `(B, H, L, 128)`. Tiling is D0's measured winner
    (q2304/m1536/k1536 on `(B, H, L, D)` bf16: 2.01x on the 4-step DiT, 55.4 dB vs an fp32
    reference). The Pallas import is inside the function so a CPU import of this module stays
    free, and `sm_scale` is the same `1/sqrt(D)` `sdpa` uses.
    """
    import jax.experimental.pallas.ops.tpu.flash_attention as flash

    length = q.shape[2]
    if length % 128:  # no legal tile (e.g. 1360x768 -> 4592 tokens): keep the XLA path
        return sdpa(q, k, v)
    block_q = tile(length, (2304, 1536, 1152, 512, 256, 128))
    block_k = tile(length, (1536, 1152, 512, 256, 128))
    return flash.flash_attention(
        q, k, v, causal=False, sm_scale=1.0 / math.sqrt(q.shape[-1]),
        block_sizes=flash.BlockSizes(block_q=block_q, block_k_major=block_k, block_k=block_k,
                                     block_b=1))


def tile(length: int, sizes: tuple[int, ...]) -> int:
    """The largest tile in `sizes` that divides `length`: the kernel rejects non-dividing tiles and
    the DiT's joint length changes with resolution (1024²: 4608 -> 2304/1536, D0's winner)."""
    return next(size for size in sizes if length % size == 0)


def conv2d(x: Array, w: Array, stride: int = 1, padding: Padding = "SAME", dilation: int = 1) -> Array:
    """NCHW->NHWC conv. `w` is HWIO. fp32 accumulation (cudnn's behaviour for bf16).

    `dilation` is the **input** dilation: `dilation - 1` zeros between input pixels.
    """
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
    """`nn.GroupNorm(32, affine=True)` over NHWC: fp32 stats, biased variance, affine in fp32."""
    n, h, w, c = x.shape
    xf = x.astype(jnp.float32).reshape(n, h * w, groups, c // groups)
    mean = jnp.mean(xf, axis=(1, 3), keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=(1, 3), keepdims=True)
    normed = ((xf - mean) * lax.rsqrt(var + eps)).reshape(n, h, w, c)
    return (normed * weight.astype(jnp.float32) + bias.astype(jnp.float32)).astype(x.dtype)


def upsample_nearest(x: Array, factor: int = 2) -> Array:
    """`F.interpolate(scale_factor=2, mode="nearest")` — an index-space broadcast, not a resize."""
    n, h, w, c = x.shape
    return jnp.broadcast_to(x[:, :, None, :, None, :], (n, h, factor, w, factor, c)).reshape(
        n, h * factor, w * factor, c
    )


def timestep_embedding(t: Array, dim: int = 256, max_period: float = 10000.0,
                       time_factor: float = 1000.0) -> Array:
    """BFL's sinusoid (`model.py:710-731`): the `1000*t` multiply is **bf16**, cos/sin are fp32.

    `t` is bf16 on entry, so `time_factor * t` rounds in bf16 before the upcast — the 0.42 %
    conditioning shift in port-plan §9.3. Do not "fix" it.
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = (t * time_factor).astype(jnp.float32)[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding.astype(t.dtype)
