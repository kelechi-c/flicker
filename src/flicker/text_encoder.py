"""Qwen3-4B text encoder, bf16, **27 of 36 layers** — we tap `hidden_states[9, 18, 27]`.

Reference: `refs/flux2/src/flux2/text_encoder.py:366-428` (which loads the FP8 checkpoint) and
`transformers` 4.56.1's `models/qwen3/modeling_qwen3.py`, which is where the numerics were read
from. Two of its cast points are *not* the DiT's:

* `Qwen3RMSNorm` — fp32 compute, cast back to bf16, **then** multiply by the bf16 weight.
* rope — `Qwen3RotaryEmbedding` returns `cos/sin` **cast to the model dtype**, so the whole
  rotation (`q*cos + rotate_half(q)*sin`) is bf16. This is the NeoX **half-split** convention,
  the opposite of the DiT's interleaved pairs.

The tokenizer is the Rust one (`tokenizers.Tokenizer.from_file`) with the chat template
hardcoded, so no `transformers` and no torch in the core path.
"""

import functools
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .config import CHAT_TEMPLATE, MAX_LENGTH, REPO, TAP_LAYERS, TOKENIZER_SUBDIR, Qwen3_4B
from .modules import qwen3_layer

Array = jax.Array
TAPS_PER_CHUNK = 9  # 27 layers / 3 taps: layers 0-8 -> h[9], 9-17 -> h[18], 18-26 -> h[27]


class Tokens(NamedTuple):
    """Always exactly `MAX_LENGTH` tokens, right-padded — padding is **kept**, never trimmed."""

    ids: Array
    mask: Array


@functools.lru_cache(maxsize=1)
def load_tokenizer() -> tuple[Tokenizer, int]:
    """Rust tokenizer + the pad id, straight from the repo — no transformers, no torch."""
    config = json.loads(
        open(hf_hub_download(REPO, f"{TOKENIZER_SUBDIR}/tokenizer_config.json")).read()
    )
    tokenizer = Tokenizer.from_file(hf_hub_download(REPO, f"{TOKENIZER_SUBDIR}/tokenizer.json"))
    # `pad_token_id` is not in the config: it stores the *string*, and the id lives in the vocab
    return tokenizer, int(tokenizer.token_to_id(config["pad_token"]))


def tokenize(prompt: str) -> Tokens:
    """Render the chat template, tokenize, right-pad/truncate to 512.

    The rendered string is `CHAT_TEMPLATE` byte-for-byte (asserted in Stage A); 12 tokens of
    overhead leaves ~500 for the prompt and longer prompts are **silently truncated**.
    """
    tokenizer, pad_id = load_tokenizer()
    ids = tokenizer.encode(CHAT_TEMPLATE.format(prompt=prompt), add_special_tokens=False).ids
    ids = ids[:MAX_LENGTH]
    mask = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
    ids = ids + [pad_id] * (MAX_LENGTH - len(ids))
    return Tokens(jnp.asarray([ids], jnp.int32), jnp.asarray([mask], jnp.int32))


def embed(p: dict[str, Array], ids: Array) -> Array:
    """`model.embed_tokens` lookup for a single batch element: (1,S) ids -> (1,S,2560)."""
    return p["embed_tokens"]["weight"][ids[0]][None]


def rope_tables(positions: Array, head_dim: int, theta: float) -> tuple[Array, Array]:
    """`Qwen3RotaryEmbedding.forward`: fp32 frequencies, `cos/sin` **cast to the model dtype**."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None]
    emb = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(emb)[None, None], jnp.sin(emb)[None, None]










def causal_mask(mask: Array) -> Array:
    """Boolean `(B,1,S,S)` keep-mask: causal **and** the padding keys removed.

    Padding positions are real tokens to the DiT, so they are computed — but they must not be
    attended to. Padding *queries* still attend to the real prefix, exactly as HF does.
    """
    length = mask.shape[1]
    causal = jnp.tril(jnp.ones((length, length), bool))
    return (mask[:, None, None, :] > 0) & causal




def trunk(p: dict[str, Array], cfg: Qwen3_4B, tokens: Tokens) -> list[Array]:
    """Run layers 0-26 in three scanned chunks and return the three tapped hidden states.

    `hidden_states[k]` is the output of layer `k-1`, so the taps are the outputs of layers
    8, 17 and 26 — one chunk boundary each, which is why the scan is split in three.
    """
    x = embed(p, tokens.ids)
    cos, sin = rope_tables(jnp.arange(MAX_LENGTH), cfg.head_dim, cfg.rope_theta)
    mask = causal_mask(tokens.mask)

    def run(x: Array, params: dict[str, Array]) -> Array:
        """Nine layers in one scan; `params` is the stacked subtree for this chunk."""
        return lax.scan(lambda h, lp: (qwen3_layer(lp, h, cos, sin, mask, cfg), None), x, params)[0]

    taps = []
    for chunk in range(len(TAP_LAYERS)):
        sl = slice(chunk * TAPS_PER_CHUNK, (chunk + 1) * TAPS_PER_CHUNK)
        chunked = jax.tree.map(lambda a, sl=sl: a[sl], p["layers"])
        taps.append(run(x, chunked))
        x = taps[-1]
    return taps


def encode(p: dict[str, Array], cfg: Qwen3_4B, tokens: Tokens) -> Array:
    """`Qwen3Embedder.forward` (`text_encoder.py:383-419`): `(1,512,7680)` = `[h9 | h18 | h27]`."""
    return jnp.concatenate(trunk(p, cfg, tokens), axis=-1)
