"""Text -> image, in three separable stages so a 16 GB chip can hold them one at a time.

Each function loads only the weights it needs, which is what makes the 1-chip TPU path (and a
24 GB L4) work: DiT 7.2 GiB + Qwen 8 GiB + VAE 0.2 GiB do **not** fit together (port-plan §7.3).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from . import config, loader, rope, sampling
from .config import Config
from .text_encoder import encode, tokenize
from .vae import load_vae, vae_decode


@dataclasses.dataclass(frozen=True)
class Paths:
    """Local file locations; the caller decides where the checkpoints live."""

    dit: str
    text_index: str
    vae: str


def latent_hw(height: int, width: int) -> tuple[int, int]:
    """The DiT works at 1/16 resolution."""
    return height // 16, width // 16


def text_ctx(paths: Paths, cfg: Config, prompt: str) -> jax.Array:
    """Prompt -> `ctx (1,512,7680)`. The 512 tokens are always full, padding included."""
    params = loader.load_text(paths.text_index, cfg.text)
    return jax.jit(encode, static_argnames=("cfg",))(params, cfg.text, tokenize(prompt))


def denoise_latent(
    paths: Paths,
    cfg: Config,
    ctx: jax.Array,
    height: int,
    width: int,
    key: jax.Array | None = None,
    noise: jax.Array | None = None,
) -> jax.Array:
    """Noise -> final latent `(1,L,128)`, four Euler steps in one scanned `jit`.

    `noise` injects a captured draw `(1,128,h,w)` — the parity path, where reproducing torch's
    CUDA RNG is infeasible (project.md §8.2 item 8) — otherwise `key` draws one in JAX.
    """
    params = loader.load_dit(paths.dit, cfg.dit)
    h, w = latent_hw(height, width)
    if noise is None:
        noise = jax.random.normal(key, (1, cfg.dit.in_channels, h, w), jnp.float32).astype(jnp.bfloat16)
    x = sampling.pack(noise)
    x_ids = jnp.asarray(rope.img_ids(h, w))
    ctx_ids = jnp.asarray(rope.txt_ids(config.MAX_LENGTH))
    steps = config.timesteps(cfg.num_steps, h * w)
    fn = jax.jit(sampling.denoise, static_argnames=("cfg", "steps"))
    return fn(params, cfg.dit, x, x_ids, ctx, ctx_ids, steps)[0]


def decode(paths: Paths, cfg: Config, latent: jax.Array, height: int, width: int) -> jax.Array:
    """Final latent -> `(1,3,H,W)` in [-1,1]."""
    params = load_vae(paths.vae, cfg.vae)
    h, w = latent_hw(height, width)
    z = sampling.unpack(latent, h, w)
    return jax.jit(vae_decode, static_argnames=("cfg",))(params, z, cfg.vae)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """`(127.5 * (x + 1)).byte()` — clamp then **truncate**, not round (project.md §8.2)."""
    clipped = np.clip(np.asarray(image, np.float32), -1.0, 1.0)
    return np.clip(127.5 * (clipped + 1.0), 0, 255).astype(np.uint8)[0].transpose(1, 2, 0)


def generate(
    paths: Paths,
    cfg: Config,
    prompt: str,
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    noise: jax.Array | None = None,
) -> np.ndarray:
    """One prompt -> one `(H,W,3)` uint8 image."""
    ctx = text_ctx(paths, cfg, prompt)
    latent = denoise_latent(paths, cfg, ctx, height, width, jax.random.key(seed), noise)
    return to_uint8(decode(paths, cfg, latent, height, width))
