"""Text -> image, in three separable stages so a 16 GB chip can hold them one at a time.

Each function loads only the weights it needs, which is what makes the 1-chip TPU path (and a
24 GB L4) work: DiT 7.75 GB + Qwen3 5.45 GB + VAE 0.1 GB is 13.3 GB of weights, and a 1024²
request peaks at about 21.5 GB on device with all of them loaded — over a 16 GiB chip.
"""

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import config, loader, rope, sampling
from .config import Config
from .text_encoder import encode, tokenize
from .vae import load_vae, vae_decode

# One trace per stage, built once at import: constructing `jax.jit(...)` per call re-pays the cache
# lookup and the trace-state bookkeeping on every request.
_encode = jax.jit(encode, static_argnames=("cfg",))
_denoise = jax.jit(sampling.denoise, static_argnames=("cfg", "steps"))
_denoise_trajectory = jax.jit(sampling.denoise_trajectory, static_argnames=("cfg", "steps"))
_vae_decode = jax.jit(vae_decode, static_argnames=("cfg",))


@dataclasses.dataclass(frozen=True)
class Paths:
    """Local file locations; the caller decides where the checkpoints live."""

    dit: str
    text_index: str
    vae: str


def latent_hw(height: int, width: int) -> tuple[int, int]:
    """The DiT works at 1/16 resolution."""
    return height // 16, width // 16


def encode_ctx(text_params: dict, cfg: Config, prompt: str | Sequence[str]) -> jax.Array:
    """Prompt -> `ctx (B,512,7680)`. The 512 tokens are always full, padding included."""
    return _encode(text_params, cfg.text, tokenize(prompt))


def denoise_from(
    dit_params: dict,
    cfg: Config,
    ctx: jax.Array,
    height: int,
    width: int,
    key: jax.Array | None = None,
    noise: jax.Array | None = None,
    return_trajectory: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Noise -> final latent `(B,L,128)`, four Euler steps in one scanned `jit`.

    `noise` injects a captured draw `(B,128,h,w)` — the parity path, where reproducing torch's
    CUDA RNG is infeasible — otherwise `key` draws one in JAX.
    When `return_trajectory=True`, returns `(final_latent, trajectory)` where `trajectory`
    has shape `(T,B,L,128)` packed bf16 post-Euler carries.
    """
    h, w = latent_hw(height, width)
    if noise is None:
        b = ctx.shape[0]
        noise = jax.random.normal(key, (b, cfg.dit.in_channels, h, w), jnp.float32).astype(jnp.bfloat16)
    x = sampling.pack(noise)
    x_ids = jnp.asarray(rope.img_ids(h, w))
    ctx_ids = jnp.asarray(rope.txt_ids(config.MAX_LENGTH))
    steps = config.timesteps(cfg.num_steps, h * w)
    if return_trajectory:
        traj = _denoise_trajectory(dit_params, cfg.dit, x, x_ids, ctx, ctx_ids, steps)
        return traj[-1], traj
    return _denoise(dit_params, cfg.dit, x, x_ids, ctx, ctx_ids, steps)[0]


def decode_from(vae_params: dict, cfg: Config, latent: jax.Array, height: int, width: int) -> jax.Array:
    """Final latent -> `(1,3,H,W)` in [-1,1]."""
    h, w = latent_hw(height, width)
    z = sampling.unpack(latent, h, w)
    return _vae_decode(vae_params, z, cfg.vae)


# The three stage loaders. On 16 GiB (v5e-1) the components do **not** co-reside — 13.3 GB of
# weights, ~21.5 GB peak with activations — so the staged functions below load, run, and drop. On 32 GiB
# (v6e-1) `load_all`/`generate_resident` keep everything alive, so weight transfers happen once at
# startup and never sit on the request path.


def text_ctx(paths: Paths, cfg: Config, prompt: str) -> jax.Array:
    return encode_ctx(loader.load_text(paths.text_index, cfg.text), cfg, prompt)


def denoise_latent(
    paths: Paths,
    cfg: Config,
    ctx: jax.Array,
    height: int,
    width: int,
    key: jax.Array | None = None,
    noise: jax.Array | None = None,
) -> jax.Array:
    return denoise_from(loader.load_dit(paths.dit, cfg.dit), cfg, ctx, height, width, key, noise)


def decode(paths: Paths, cfg: Config, latent: jax.Array, height: int, width: int) -> jax.Array:
    return decode_from(load_vae(paths.vae, cfg.vae), cfg, latent, height, width)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """`(127.5 * (x + 1)).byte()` — clamp then **truncate**, not round."""
    clipped = np.clip(np.asarray(image, np.float32), -1.0, 1.0)
    arr = np.clip(127.5 * (clipped + 1.0), 0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0].transpose(1, 2, 0)
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr


def generate(
    paths: Paths,
    cfg: Config,
    prompt: str,
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    noise: jax.Array | None = None,
) -> np.ndarray:
    """One prompt -> one `(H,W,3)` uint8 image, staging the weights (the 16 GiB shape)."""
    ctx = text_ctx(paths, cfg, prompt)
    latent = denoise_latent(paths, cfg, ctx, height, width, jax.random.key(seed), noise)
    return to_uint8(decode(paths, cfg, latent, height, width))


@dataclasses.dataclass(frozen=True)
class Resident:
    """Every component loaded at once — the 32 GiB (v6e-1) serving shape."""

    text: dict
    dit: dict
    vae: dict


def load_all(paths: Paths, cfg: Config) -> Resident:
    """Load all three components and keep them. ~13.3 GB of weights: fits v6e-1, not v5e-1."""
    return Resident(
        text=loader.load_text(paths.text_index, cfg.text),
        dit=loader.load_dit(paths.dit, cfg.dit),
        vae=load_vae(paths.vae, cfg.vae),
    )


def generate_resident(
    resident: Resident,
    cfg: Config,
    prompt: str | Sequence[str],
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    noise: jax.Array | None = None,
) -> np.ndarray:
    """One or more prompts with nothing left to load — the second and subsequent requests of a server.

    The difference against `generate` is not arithmetic, it is the absence of ~13 GB of device
    transfers on the critical path.
    """
    ctx = encode_ctx(resident.text, cfg, prompt)
    latent = denoise_from(resident.dit, cfg, ctx, height, width, jax.random.key(seed), noise)
    return to_uint8(decode_from(resident.vae, cfg, latent, height, width))
