import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import config, loader, rope, sampling
from .config import Config
from .text_encoder import encode, tokenize
from .vae import load_vae, vae_decode

_encode = jax.jit(encode, static_argnames=("cfg",))
_denoise = jax.jit(sampling.denoise, static_argnames=("cfg", "steps"))
_denoise_trajectory = jax.jit(sampling.denoise_trajectory, static_argnames=("cfg", "steps"))
_vae_decode = jax.jit(vae_decode, static_argnames=("cfg",))


@dataclasses.dataclass(frozen=True)
class Paths:

    dit: str
    text_index: str
    vae: str


def latent_hw(height: int, width: int) -> tuple[int, int]:
    return height // 16, width // 16


def encode_ctx(text_params: dict, cfg: Config, prompt: str | Sequence[str]) -> jax.Array:
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
    h, w = latent_hw(height, width)
    z = sampling.unpack(latent, h, w)
    return _vae_decode(vae_params, z, cfg.vae)


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
    clipped = np.clip(np.asarray(image, np.float32), -1.0, 1.0)
    arr = np.clip(127.5 * (clipped + 1.0), 0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0].transpose(1, 2, 0)
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr


# loads one stage at a time, for accelerators too small to hold everything
def generate(
    paths: Paths,
    cfg: Config,
    prompt: str,
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    noise: jax.Array | None = None,
) -> np.ndarray:
    ctx = text_ctx(paths, cfg, prompt)
    latent = denoise_latent(paths, cfg, ctx, height, width, jax.random.key(seed), noise)
    return to_uint8(decode(paths, cfg, latent, height, width))


@dataclasses.dataclass(frozen=True)
class Resident:

    text: dict
    dit: dict
    vae: dict


# every stage resident on device (~13 GB)
def load_all(paths: Paths, cfg: Config) -> Resident:
    return Resident(
        text=loader.load_text(paths.text_index, cfg.text),
        dit=loader.load_dit(paths.dit, cfg.dit),
        vae=load_vae(paths.vae, cfg.vae),
    )


# text -> denoise -> decode with all weights already on device
def generate_resident(
    resident: Resident,
    cfg: Config,
    prompt: str | Sequence[str],
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    noise: jax.Array | None = None,
) -> np.ndarray:
    ctx = encode_ctx(resident.text, cfg, prompt)
    latent = denoise_from(resident.dit, cfg, ctx, height, width, jax.random.key(seed), noise)
    return to_uint8(decode_from(resident.vae, cfg, latent, height, width))
