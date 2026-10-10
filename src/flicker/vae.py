import jax
import jax.numpy as jnp

from .config import VAE
from .loader import header, load_flat
from .modules import attn_block, conv, gn, resnet_block, unpatchify, upsample
from .nn import conv2d

Array = jax.Array
Params = dict[str, Array]

DECODER = "decoder"
UP_BLOCKS = "decoder.up_blocks"

IGNORED = ("bn.num_batches_tracked",)
ENCODER_PREFIXES = ("encoder.", "quant_conv.")
ENCODER_TENSORS = 108


def _resnet_keys(shortcut: bool) -> set[str]:
    keys = {
        "norm1.weight", "norm1.bias", "conv1.weight", "conv1.bias",
        "norm2.weight", "norm2.bias", "conv2.weight", "conv2.bias",
    }
    if shortcut:
        keys |= {"conv_shortcut.weight", "conv_shortcut.bias"}
    return keys


def _attn_keys() -> set[str]:
    return {
        "group_norm.weight", "group_norm.bias",
        "to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias",
        "to_v.weight", "to_v.bias", "to_out.0.weight", "to_out.0.bias",
    }


def vae_keys(cfg: VAE) -> set[str]:
    levels, resnets = len(cfg.ch_mult), cfg.num_res_blocks + 1
    keys = {"bn.running_mean", "bn.running_var", "post_quant_conv.weight", "post_quant_conv.bias"}
    keys |= {f"decoder.{k}" for k in (
        "conv_in.weight", "conv_in.bias", "conv_norm_out.weight", "conv_norm_out.bias",
        "conv_out.weight", "conv_out.bias",
    )}
    keys |= {f"decoder.mid_block.attentions.0.{k}" for k in _attn_keys()}
    keys |= {f"decoder.mid_block.resnets.{j}.{k}" for j in range(2) for k in _resnet_keys(False)}
    out_ch = [cfg.ch * cfg.ch_mult[levels - 1 - i] for i in range(levels)]
    for i in range(levels):
        in_ch = cfg.ch * cfg.ch_mult[-1] if i == 0 else out_ch[i - 1]
        for j in range(resnets):
            width = in_ch if j == 0 else out_ch[i]
            keys |= {f"decoder.up_blocks.{i}.resnets.{j}.{k}" for k in _resnet_keys(width != out_ch[i])}
        if i < levels - 1:
            keys |= {f"decoder.up_blocks.{i}.upsamplers.0.conv.{k}" for k in ("weight", "bias")}
    return keys


def _group(flat: Params, prefix: str) -> Params:
    return {k[len(prefix) + 1 :]: v for k, v in flat.items() if k.startswith(prefix + ".")}


def load_vae(path: str, cfg: VAE) -> Params:
    ignored = tuple(k for k in header(path) if k.startswith(ENCODER_PREFIXES))
    assert len(ignored) == ENCODER_TENSORS, f"{ENCODER_TENSORS} encoder tensors expected, {len(ignored)} found"
    flat = load_flat(path, vae_keys(cfg), ignore=IGNORED + ignored)
    decoder = {k: v for k, v in _group(flat, DECODER).items() if not k.startswith("up_blocks.")}
    return jax.device_put({
        "bn": _group(flat, "bn"),
        "post_quant_conv": _group(flat, "post_quant_conv"),
        DECODER: decoder,
        UP_BLOCKS: _group(flat, UP_BLOCKS),
    })


def _nhwc(x: Array) -> Array:
    return jnp.transpose(x, (0, 2, 3, 1))


def _nchw(x: Array) -> Array:
    return jnp.transpose(x, (0, 3, 1, 2))


def _put(trace: dict[str, Array] | None, key: str, x: Array) -> None:
    if trace is not None:
        trace[key] = x


def decoder_forward(p: Params, x: Array, cfg: VAE, trace: dict[str, Array] | None = None) -> Array:
    d, up = p[DECODER], p[UP_BLOCKS]
    h = conv(_nhwc(x), d, "conv_in")
    _put(trace, "dec/conv_in", _nchw(h))
    h = resnet_block(h, d, "mid_block.resnets.0", cfg)
    h = attn_block(h, d, "mid_block.attentions.0", cfg)
    h = resnet_block(h, d, "mid_block.resnets.1", cfg)
    _put(trace, "dec/mid", _nchw(h))
    for i in range(len(cfg.ch_mult)):
        for j in range(cfg.num_res_blocks + 1):
            h = resnet_block(h, up, f"{i}.resnets.{j}", cfg)
        if i < len(cfg.ch_mult) - 1:
            h = upsample(h, up, f"{i}.upsamplers.0", cfg)
        _put(trace, f"dec/up/{i}", _nchw(h))
    h = conv(jax.nn.silu(gn(h, d, "conv_norm_out", cfg)), d, "conv_out")
    _put(trace, "dec/out", _nchw(h))
    return _nchw(h)


def _round_bf16(x: Array) -> Array:
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def inv_normalize(p: Params, z: Array, cfg: VAE) -> Array:
    var, mean = (p["bn"][k].astype(jnp.float32).reshape(1, -1, 1, 1)
                 for k in ("running_var", "running_mean"))
    std = _round_bf16(jnp.sqrt(_round_bf16(var + cfg.bn_eps)))
    return (_round_bf16(z.astype(jnp.float32) * std) + mean).astype(z.dtype)


# batch-norm un-normalize, unpatchify, then the conv decoder
def vae_decode(p: Params, z: Array, cfg: VAE, trace: dict[str, Array] | None = None) -> Array:
    _put(trace, "unpatch", z)
    z_in = inv_normalize(p, z, cfg)
    _put(trace, "z_in", z_in)
    pq = p["post_quant_conv"]
    x = conv2d(_nhwc(unpatchify(z_in, cfg.patch)), pq["weight"], padding="VALID") + pq["bias"]
    return decoder_forward(p, _nchw(x), cfg, trace)
