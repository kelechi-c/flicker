"""The entire "framework": ~10 pure functions, no flax/linen/nnx.

Every one of these is a place where the dtype had to be chosen deliberately, which is why they
are hand-written rather than imported (`docs/port-plan.md` §1). Param leaves are torch-named:
`linear` takes `(in, out)` (transposed at load), `conv2d` takes `(h, w, in, out)` (OIHW -> HWIO).
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Padding = str
Attention = Callable[[Array, Array, Array], Array]


def linear(x: Array, w: Array) -> Array:
    """`nn.Linear` with bias=False. Torch accumulates bf16 GEMMs in fp32, as do XLA's backends."""
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


def conv2d(x: Array, w: Array, stride: int = 1, padding: Padding = "SAME") -> Array:
    """NCHW->NHWC conv. `w` is HWIO. fp32 accumulation (cudnn's behaviour for bf16)."""
    out = lax.conv_general_dilated(
        x,
        w,
        (stride, stride),
        padding.upper(),
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
