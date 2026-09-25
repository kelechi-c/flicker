import jax
import jax.numpy as jnp
from jax import lax

from .config import Klein4B
from .modules import double_block, last_layer, mlp_embedder, modulation, single_block
from .nn import linear, timestep_embedding
from .rope import embed_nd

Array = jax.Array


def block_params(p: dict[str, Array], key: str) -> dict[str, Array]:
    return p[key]


# t2i velocity prediction; x (1,L,128), ctx (1,512,7680), t - (B,)
def forward(
    p: dict[str, Array],
    cfg: Klein4B,
    x: Array,
    x_ids: Array,
    t: Array,
    ctx: Array,
    ctx_ids: Array,
) -> Array:
    vec = mlp_embedder(p["time_in"], timestep_embedding(t, 256))
    mod_img = modulation(p["double_stream_modulation_img"], vec, double=True)
    mod_txt = modulation(p["double_stream_modulation_txt"], vec, double=True)
    mod_single = modulation(p["single_stream_modulation"], vec, double=False)

    img = linear(x, p["img_in"]["weight"])
    txt = linear(ctx, p["txt_in"]["weight"])
    pe_x = embed_nd(x_ids, cfg.axes_dim, cfg.theta)
    pe_ctx = embed_nd(ctx_ids, cfg.axes_dim, cfg.theta)

    def one_double(carry: tuple[Array, Array], block: dict[str, Array]) -> tuple[Array, Array]:
        img, txt = carry
        return double_block(block, img, txt, pe_x, pe_ctx, mod_img, mod_txt, cfg.num_heads), None

    (img, txt), _ = lax.scan(one_double, (img, txt), block_params(p, "double_blocks"), unroll=cfg.block_unroll)

    seq = jnp.concatenate([txt, img], axis=1)
    pe = jnp.concatenate([pe_ctx, pe_x], axis=2)
    num_txt = ctx.shape[1]

    def one_single(x: Array, block: dict[str, Array]) -> tuple[Array, None]:
        return single_block(block, x, pe, mod_single, cfg.num_heads), None

    seq, _ = lax.scan(one_single, seq, block_params(p, "single_blocks"), unroll=cfg.block_unroll)
    return last_layer(p["final_layer"], seq[:, num_txt:], vec)