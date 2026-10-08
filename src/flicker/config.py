""" 
constants, repo ids, etc.
FLUX.2 [klein] 4B model configs source: `flux2/src/flux2/model.py:38-49`. 
"""

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

# the chat template renders to exactly this when enable_thinking=False. byte-exact: the think
# tags are plain ascii text, **not** special tokens (the `<|im_start|>` family are).
# the rendered truth is `<think>` / `</think>`, as `apply_chat_template` produces it.
CHAT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

MAX_LENGTH = 512  # text is always right-padded to this; the DiT never masks it
TAP_LAYERS = (9, 18, 27)  # hidden_states indices = outputs of decoder layers 8/17/26

Dtype = Literal["bfloat16", "float32"]
BF16 = jnp.bfloat16


@dataclass(frozen=True)
class Klein4B:
    """DiT shape. `theta` is the rope base (not a time constant); head_dim = 3072/24 = 128."""

    in_channels: int = 128
    context_in_dim: int = 7680  # 3 (chosen layers) x 2560 (qwen3 hidden state), stacked
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    block_unroll: int = 1  # static scan variant; 1 preserves the original graph
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False
    # static: it selects a traced kernel, so it must be part of the jit's cache key (a rebind of
    # a module global is silently ignored by the trace cache). "bounded" is `kernels.attend`:
    # flash minus the running max, valid while `kernels.logit_bound` of every block stays under
    # `kernels.EXP_LIMIT` (`loader.load_dit` checks). "flash" is the library kernel; both fall
    # back to "sdpa" off TPU.
    attention: Literal["sdpa", "flash", "bounded"] = "bounded"
    # static for the same reason: "kernel" fuses the head transpose, QK RMSNorm and rope into one
    # Pallas pass (`kernels.qk_prep`), "xla" keeps the three separate ops. On CPU always "xla".
    qk_prep: Literal["xla", "kernel"] = "kernel"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

# only taps features form 3 layers, last layer being 27 (no need to run layers 28-36)
@dataclass(frozen=True)
class Qwen3_4B:
    hidden_size: int = 2560
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8  # GQA
    head_dim: int = 128
    intermediate: int = 9728
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    vocab: int = 151936

    @property
    def layers_needed(self) -> int:
        return max(TAP_LAYERS)  # 27: hidden_states[k] = output of layer k-1

# diffusers `AutoencoderKLFlux2` decoder
@dataclass(frozen=True)
class VAE:
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 32
    patch: int = 2  # ps = [2,2]: 32ch at H/8 -> 128ch at H/16
    bn_eps: float = 1e-4
    groups: int = 32
    norm_eps: float = 1e-6
    # static: "dilated" runs nearest-2x + 3x3 as one 4x4 conv on the zero-inserted input (the
    # default: 37.4 -> 26.1 ms per 1024² decode on v6e); "nearest" is the reference's literal form.
    upsample: Literal["nearest", "dilated"] = "dilated"


def timesteps(num_steps: int, seq_len: int) -> tuple[float, ...]:
    """BFL's schedule (`sampling.py:240-266`) on the host, in **float32** like the reference.

    `torch.linspace(1, 0, n)` is float32 and `generalized_time_snr_shift` runs on it, so the
    schedule values are float32-rounded before `torch.full(..., dtype=bf16)` rounds them again.
    `t=0` is special-cased: torch tensor division yields `inf` -> `0.0`, python float division
    raises.
    """
    exp_mu = np.float32(math.exp(empirical_mu(seq_len, num_steps)))
    grid = np.linspace(np.float32(1.0), np.float32(0.0), num_steps + 1, dtype=np.float32)
    # a tuple, not a list: it is passed to `jax.jit` as a static argument
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

# main flicker pipeline config
@dataclass(frozen=True)
class Config:
    dit: Klein4B = field(default_factory=Klein4B)
    text: Qwen3_4B = field(default_factory=Qwen3_4B)
    vae: VAE = field(default_factory=VAE)
    num_steps: int = 4
    guidance: float = 1.0
    dtype: Dtype = "bfloat16"