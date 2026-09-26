# Qwen3-4B text encoder, bf16, **27 of 36 layers**. tap `hidden_states[9, 18, 27]`.
# Reference: `flux2/src/flux2/text_encoder.py:366-428` (which loads the FP8 checkpoint)

import functools
import json
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .config import CHAT_TEMPLATE, MAX_LENGTH, REPO, TAP_LAYERS, TOKENIZER_SUBDIR, Qwen3_4B
from .modules import qwen3_layer

Array = jax.Array
ArrayLike = Array | np.ndarray  # host ids/mask are converted by the jit boundary, not eagerly
TOKENIZER_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


class Tokens(NamedTuple):
    ids: ArrayLike
    mask: ArrayLike


@functools.lru_cache(maxsize=1)
def load_tokenizer(revision: str | None = None) -> tuple[Tokenizer, int]:
    rev = revision or TOKENIZER_REVISION
    kwargs = {"revision": rev} if rev else {}
    cfg_file = hf_hub_download(REPO, f"{TOKENIZER_SUBDIR}/tokenizer_config.json", **kwargs)
    tok_file = hf_hub_download(REPO, f"{TOKENIZER_SUBDIR}/tokenizer.json", **kwargs)
    config = json.loads(open(cfg_file).read())
    tokenizer = Tokenizer.from_file(tok_file)
    # `pad_token_id` is not in the config: it stores the *string*, and the id lives in the vocab.
    return tokenizer, int(tokenizer.token_to_id(config["pad_token"]))


def tokenize(prompt: str | Sequence[str]) -> Tokens:
    tokenizer, pad_id = load_tokenizer()
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    all_ids, all_masks = [], []
    for p in prompts:
        ids = tokenizer.encode(CHAT_TEMPLATE.format(prompt=p), add_special_tokens=False).ids[:MAX_LENGTH]
        mask = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
        ids = ids + [pad_id] * (MAX_LENGTH - len(ids))
        all_ids.append(ids)
        all_masks.append(mask)
    return Tokens(np.asarray(all_ids, np.int32), np.asarray(all_masks, np.int32))


# `model.embed_tokens` lookup: (B,S) ids -> (B,S,2560)
def embed(p: dict[str, Array], ids: ArrayLike) -> Array:
    return p["embed_tokens"]["weight"][ids]


# `Qwen3RotaryEmbedding.forward`: fp32 frequencies, `cos/sin`
def rope_tables(positions: Array, head_dim: int, theta: float) -> tuple[Array, Array]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None]
    emb = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(emb)[None, None], jnp.sin(emb)[None, None]


def causal_mask(mask: ArrayLike) -> Array:
    length = mask.shape[1]
    causal = jnp.tril(jnp.ones((length, length), bool))
    return (mask[:, None, None, :] > 0) & causal


# One `lax.scan` over layers 0-26; the body returns its output as the carry and as `ys`, so the
# taps are plain slices of `ys`. `hidden_states[k]` is the output of layer `k-1`, so tap `k` is
# `ys[k-1]` (9/18/27 -> the outputs of layers 8/17/26). Three 9-layer scans cost 2.3x this on TPU.
def trunk(p: dict[str, Array], cfg: Qwen3_4B, tokens: Tokens) -> list[Array]:
    x = embed(p, tokens.ids)
    cos, sin = rope_tables(jnp.arange(MAX_LENGTH), cfg.head_dim, cfg.rope_theta)
    mask = causal_mask(tokens.mask)

    def body(h: Array, lp: dict[str, Array]) -> tuple[Array, Array]:
        y = qwen3_layer(lp, h, cos, sin, mask, cfg)
        return y, y

    ys = lax.scan(body, x, p["layers"])[1]
    return [ys[k - 1] for k in TAP_LAYERS]


# `Qwen3Embedder.forward` (`text_encoder.py:383-419`)
def encode(p: dict[str, Array], cfg: Qwen3_4B, tokens: Tokens) -> Array:
    return jnp.concatenate(trunk(p, cfg, tokens), axis=-1)
