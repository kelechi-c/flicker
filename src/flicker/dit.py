import math

import jax
import jax.numpy as jnp
from jax import lax

from .config import Klein4B
from .kernels import attend as bounded_attend
from .kernels import lane_pair
from .modules import double_block, last_layer, mlp_embedder, modulation, single_block
from .nn import flash_sdpa, linear, sdpa, timestep_embedding
from .rope import embed_nd

Array = jax.Array


def block_params(p: dict[str, Array], key: str) -> dict[str, Array]:
    return p[key]


# one t2i velocity prediction: modulation -> 5 double blocks -> 20 single blocks (both lax.scan) -> final layer
def forward(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    t: Array,
    ctx: Array,
    ctx_ids: Array,
) -> Array:
    on_tpu = jax.default_backend() == "tpu"
    # the bounded kernel needs the fused qk-prep and 128-multiple streams; otherwise fall back to flash/sdpa
    bounded = (cfg.attention == "bounded" and cfg.qk_prep == "kernel" and on_tpu
               and x.shape[1] % 128 == 0 and ctx.shape[1] % 128 == 0)
    attend = bounded_attend if bounded else flash_sdpa if cfg.attention != "sdpa" and on_tpu else sdpa
    q_gain = 1.0 / math.sqrt(cfg.head_dim) if bounded else 1.0
    vec = mlp_embedder(p["time_in"], timestep_embedding(t, 256))
    mod_img = modulation(p["double_stream_modulation_img"], vec, double=True)
    mod_txt = modulation(p["double_stream_modulation_txt"], vec, double=True)
    mod_single = modulation(p["single_stream_modulation"], vec, double=False)

    img = linear(x, p["img_in"]["weight"])
    txt = linear(ctx, p["txt_in"]["weight"])
    pe_x = embed_nd(x_ids, cfg.axes_dim, cfg.theta)
    pe_ctx = embed_nd(ctx_ids, cfg.axes_dim, cfg.theta)
    pe = jnp.concatenate([pe_ctx, pe_x], axis=2)
    freqs = lane_pair(pe, q_gain) if cfg.qk_prep == "kernel" and on_tpu else None

    def one_double(carry: tuple[Array, Array], block: dict[str, Array]) -> tuple[Array, Array]:
        img, txt = carry
        return double_block(block, img, txt, pe_x, pe_ctx, mod_img, mod_txt, cfg.num_heads,
                            attend, freqs), None

    (img, txt), _ = lax.scan(one_double, (img, txt), block_params(p, "double_blocks"), unroll=cfg.block_unroll)

    seq = jnp.concatenate([txt, img], axis=1)
    num_txt = ctx.shape[1]

    def one_single(x: Array, block: dict[str, Array]) -> tuple[Array, None]:
        return single_block(block, x, pe, mod_single, cfg.num_heads, attend, freqs), None

    seq, _ = lax.scan(one_single, seq, block_params(p, "single_blocks"), unroll=cfg.block_unroll)
    return last_layer(p["final_layer"], seq[:, num_txt:], vec)
