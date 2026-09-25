"""4-axis rope, the position-id conventions, and latent packing.

The single subtlest part of the port (`project.md` §3.4). Three things must all be right:
interleaved **adjacent-pair** rotation (not Llama's half-split), pairs `0-15=t, 16-31=h, 32-47=w,
48-63=l`, and fp32-in / bf16-out.
"""

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
FloatArray = np.ndarray


def rope_freqs(pos: Array, dim: int, theta: float = 2000.0) -> Array:
    """`rope` (`model.py:818-825`) for one axis: (..., n) -> (..., n, dim/2, 2, 2) fp32.

    The last `(2,2)` is `[[cos, -sin], [sin, cos]]` per frequency, which `apply_rope` reads as
    the two rotation coefficients. `omega` is built explicitly in fp32 (JAX has no int true
    division).
    """
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta**scale)
    angles = pos.astype(jnp.float32)[..., None] * omega
    stacked = jnp.stack([jnp.cos(angles), -jnp.sin(angles), jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return stacked.reshape(*angles.shape, 2, 2)


def embed_nd(ids: Array, axes_dim: tuple[int, ...] = (32, 32, 32, 32), theta: float = 2000.0) -> Array:
    """`EmbedND` (`model.py:694-707`): (B, N, 4) ids -> (B, 1, N, 64, 2, 2) fp32.

    Axes are concatenated on the frequency axis, then a **head** axis of size 1 is inserted so
    the table broadcasts over all 24 heads.
    """
    if ids.ndim == 2:
        ids = ids[None]
    per_axis = [rope_freqs(ids[..., i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return jnp.concatenate(per_axis, axis=-3)[:, None]


def apply_rope(xq: Array, xk: Array, freqs: Array) -> tuple[Array, Array]:
    """`apply_rope` (`model.py:828-833`): rotate adjacent pairs in fp32, cast back to bf16.

    `out[2d] = cos*x[2d] - sin*x[2d+1]`, `out[2d+1] = sin*x[2d] + cos*x[2d+1]`.
    """
    def rotate(x: Array) -> Array:
        pairs = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 1, 2)
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.reshape(*x.shape).astype(x.dtype)

    return rotate(xq), rotate(xk)


def img_ids(height: int, width: int, t: float = 0.0) -> FloatArray:
    """`batched_prc_img` (`sampling.py:141-151`): (1, h*w, 4) ids `(t, h, w, 0)`, row-major in
    `h,w`. Batched because the DiT is batch-1 and an unbatched id table silently broadcasts
    wrong — the batch axis is part of the contract.

    `cartesian_prod(t,h,w,l)` is t-slowest/l-fastest, which is the same order as
    `rearrange("c h w -> (h w) c")` — the token order the latent is packed in.
    """
    flat = np.arange(height * width)
    ids = np.stack(
        [np.full(height * width, t), flat // width, flat % width, np.zeros(height * width)], axis=-1
    ).astype(np.float32)
    return ids[None]


def txt_ids(length: int) -> FloatArray:
    """`batched_prc_txt` (`sampling.py:93-103`): (1, length, 4). Text tokens vary only on the `l`
    axis — disjoint from image tokens, which vary only on `h,w`."""
    ids = np.stack(
        [np.zeros(length), np.zeros(length), np.zeros(length), np.arange(length)], axis=-1
    ).astype(np.float32)
    return ids[None]
