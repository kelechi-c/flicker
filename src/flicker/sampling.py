"""Rectified-flow sampling: the schedule (host floats), the Euler scan, and latent packing.

The schedule is computed on the host in python float64 (`config.timesteps`) and rounded to bf16
with ml_dtypes so the rounding is **fp64 -> bf16 in one step**, exactly like
`torch.full((B,), t_curr, dtype=torch.bfloat16)`. Doing it through fp32 would double-round.
"""

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
    """Round host float64 values to bf16 in a single step (numpy does not go via fp32)."""
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
    """One production Euler update: `pred` then `x + delta * pred`, all bf16.

    `t` and `delta` are the bf16 scalars the schedule rounds to; `t` is broadcast over the
    batch. Returns `(x_new, pred)` so the two scan loops below share this arithmetic exactly
    and cannot drift — `denoise` keeps `x_new` as its carry, `denoise_trajectory` also emits it.
    """
    pred = forward(p, cfg, x, x_ids, jnp.broadcast_to(t, (x.shape[0],)), ctx, ctx_ids)
    return x + delta * pred, pred


def _schedule(steps: Sequence[float]) -> tuple[Array, Array]:
    """Host floats -> the bf16 `(t_curr, dt)` scan inputs, one entry per Euler update."""
    t_curr = bf16(steps[:-1])
    dt = bf16([prev - cur for cur, prev in pairwise(steps)])
    return t_curr, dt


def denoise(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    ctx: Array,
    ctx_ids: Array,
    steps: Sequence[float],
) -> tuple[Array, Array]:
    """The 4-step distilled Euler loop (`sampling.py:269-307`), as a `lax.scan`.

    `x += (t_prev - t_curr) * pred` with the step size and the timestep both **bf16** — the
    schedule is descending (t=1 is noise) and there are exactly `len(steps) - 1` evaluations.
    Returns the final latent and the stacked per-step predictions (a ladder gate).

    `steps` is static (python floats), so it is a compile-time constant of the scan length.
    """
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
    """Observer variant of `denoise`: the post-Euler carries `(T, B, L, 128)` bf16.

    Same single `lax.scan`, same `euler` arithmetic; only the scan's `ys` differ. Entry `t` is
    `x` **after** Euler update `t` (x0..x3 for the 4-step schedule), in the **packed** layout the
    official `latents/{pid}-seed{k}.safetensors` stores (`step0..step3`), so it is saved as-is;
    `result[-1]` equals `denoise(...)[0]` bit-for-bit. Unpack (`sampling.unpack`) only for VAE
    decode. Separate from `denoise` so production timing never pays for four live latents.
    """
    def step(x: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        x_new, _ = euler(p, cfg, x, x_ids, ctx, ctx_ids, *inputs)
        return x_new, x_new

    return lax.scan(step, x, _schedule(steps))[1]


def pack(latent: Array) -> Array:
    """(B,128,h,w) -> (B,h*w,128): the token order `rearrange("c h w -> (h w) c")` produces."""
    b, c, h, w = latent.shape
    return latent.transpose(0, 2, 3, 1).reshape(b, h * w, c)


def unpack(x: Array, height: int, width: int) -> Array:
    """`scatter_ids` for the single-image t2i case: (B,L,128) -> (B,128,h,w).

    The general function scatters into a `t*h*w` buffer keyed by compressed time ids; with one
    image `t=1` and `flat_ids == arange(h*w)`, so it degenerates to a pure reshape.
    """
    b, length, c = x.shape
    assert length == height * width, f"{length} tokens do not tile {height}x{width}"
    return x.reshape(b, height, width, c).transpose(0, 3, 1, 2)
