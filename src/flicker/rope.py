import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
FloatArray = np.ndarray


def rope_freqs(pos: Array, dim: int, theta: float = 2000.0) -> Array:
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta**scale)
    angles = pos.astype(jnp.float32)[..., None] * omega
    stacked = jnp.stack([jnp.cos(angles), -jnp.sin(angles), jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return stacked.reshape(*angles.shape, 2, 2)


def embed_nd(ids: Array, axes_dim: tuple[int, ...] = (32, 32, 32, 32), theta: float = 2000.0) -> Array:
    if ids.ndim == 2:
        ids = ids[None]
    per_axis = [rope_freqs(ids[..., i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return jnp.concatenate(per_axis, axis=-3)[:, None]


def apply_rope(xq: Array, xk: Array, freqs: Array) -> tuple[Array, Array]:
    def rotate(x: Array) -> Array:
        pairs = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 1, 2)
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.reshape(*x.shape).astype(x.dtype)

    return rotate(xq), rotate(xk)


def img_ids(height: int, width: int, t: float = 0.0) -> FloatArray:
    flat = np.arange(height * width)
    ids = np.stack(
        [np.full(height * width, t), flat // width, flat % width, np.zeros(height * width)], axis=-1
    ).astype(np.float32)
    return ids[None]


def txt_ids(length: int) -> FloatArray:
    ids = np.stack(
        [np.zeros(length), np.zeros(length), np.zeros(length), np.arange(length)], axis=-1
    ).astype(np.float32)
    return ids[None]
