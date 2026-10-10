from collections.abc import Sequence
from itertools import pairwise

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

from .config import Klein4B
from .dit import forward

Array = jax.Array


def bf16(values: Sequence[float]) -> Array:
    return jnp.asarray(np.asarray(values, dtype=ml_dtypes.bfloat16))


def euler(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    ctx: Array,
    ctx_ids: Array,
    t: Array,
    delta: Array,
) -> tuple[Array, Array]:
    pred = forward(p, cfg, x, x_ids, jnp.broadcast_to(t, (x.shape[0],)), ctx, ctx_ids)
    return x + delta * pred, pred


def _schedule(steps: Sequence[float]) -> tuple[Array, Array]:
    t_curr = bf16(steps[:-1])
    dt = bf16([prev - cur for cur, prev in pairwise(steps)])
    return t_curr, dt


# 4-step euler over the shifted schedule
def denoise(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    ctx: Array,
    ctx_ids: Array,
    steps: Sequence[float],
) -> tuple[Array, Array]:
    def step(x: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        return euler(p, cfg, x, x_ids, ctx, ctx_ids, *inputs)

    return lax.scan(step, x, _schedule(steps))


def denoise_trajectory(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    ctx: Array,
    ctx_ids: Array,
    steps: Sequence[float],
) -> Array:
    def step(x: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        x_new, _ = euler(p, cfg, x, x_ids, ctx, ctx_ids, *inputs)
        return x_new, x_new

    return lax.scan(step, x, _schedule(steps))[1]


def pack(latent: Array) -> Array:
    b, c, h, w = latent.shape
    return latent.transpose(0, 2, 3, 1).reshape(b, h * w, c)


def unpack(x: Array, height: int, width: int) -> Array:
    b, length, c = x.shape
    assert length == height * width, f"{length} tokens do not tile {height}x{width}"
    return x.reshape(b, height, width, c).transpose(0, 3, 1, 2)
