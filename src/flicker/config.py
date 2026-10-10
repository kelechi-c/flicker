import math
from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
import numpy as np

REPO = "black-forest-labs/FLUX.2-klein-4B"
DIT_FILE = "flux-2-klein-4b.safetensors"
TEXT_SUBDIR = "text_encoder"
TOKENIZER_SUBDIR = "tokenizer"
VAE_SUBDIR = "vae"

CHAT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

MAX_LENGTH = 512
TAP_LAYERS = (9, 18, 27)

Dtype = Literal["bfloat16", "float32"]
BF16 = jnp.bfloat16


@dataclass(frozen=True)
class Klein4B:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    block_unroll: int = 1
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False
    attention: Literal["sdpa", "flash", "bounded"] = "bounded"
    qk_prep: Literal["xla", "kernel"] = "kernel"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


@dataclass(frozen=True)
class Qwen3_4B:
    hidden_size: int = 2560
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 9728
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    vocab: int = 151936

    @property
    def layers_needed(self) -> int:
        return max(TAP_LAYERS)


@dataclass(frozen=True)
class VAE:
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 32
    patch: int = 2
    bn_eps: float = 1e-4
    groups: int = 32
    norm_eps: float = 1e-6
    upsample: Literal["nearest", "dilated"] = "dilated"


def timesteps(num_steps: int, seq_len: int) -> tuple[float, ...]:
    exp_mu = np.float32(math.exp(empirical_mu(seq_len, num_steps)))
    grid = np.linspace(np.float32(1.0), np.float32(0.0), num_steps + 1, dtype=np.float32)
    return tuple(_shift(float(t), exp_mu) for t in grid)


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    return a * num_steps + (m_200 - 200.0 * a)


def _shift(t: float, exp_mu: np.float32, sigma: float = 1.0) -> float:
    if t == 0.0:
        return 0.0
    return float(
        exp_mu / (exp_mu + (np.float32(1.0) / np.float32(t) - np.float32(1.0)) ** np.float32(sigma))
    )

@dataclass(frozen=True)
class Config:
    dit: Klein4B = field(default_factory=Klein4B)
    text: Qwen3_4B = field(default_factory=Qwen3_4B)
    vae: VAE = field(default_factory=VAE)
    num_steps: int = 4
    guidance: float = 1.0
    dtype: Dtype = "bfloat16"
