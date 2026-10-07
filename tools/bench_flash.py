"""why does flash attention run at 269 tf/s? an ablation on one colab v6e-1, no weights, no token.

one file for `colab exec -s <session> -f tools/bench_flash.py` (it forwards no argv): the first call starts
a detached chain, every call prints a status json, re-exec to poll.

every arm is one attention call on synthetic dit-shaped q/k/v (`(1, 24, 4608, 128)` bf16, q/k rms-normed
like qk-norm leaves them), timed as CHAIN calls in one `fori_loop` (v carries the output, so no call can be
hoisted or deduplicated) and reported per call with its snr against an fp32 `softmax(qk^t) v`.

- `lib`: the shipped kernel (`jax...flash_attention`, q2304 / k1536) — the 269 tf/s to explain.
- `online`: our kernel with the library's arithmetic — proves the harness reproduces `lib`.
- `dots`: the two matmuls only, no softmax — what the mxu alone costs in this tiling.
- `free`: softmax without the running max (`exp(s)`, one sum, one divide at the end). legal only while
  `|logit| < 88`; qk-norm bounds it (see docs/optimization-notes.md 2026-10-07).
- `round`: `free` normalised by the bf16-rounded weights the matmul sees. this is `flicker.kernels.attend`.
- `*_pre`: `1/sqrt(d)` folded into q beforehand, so the score tile is not rescaled.
- `free_pre` over a tile sweep, and `merged`: `free_pre` reading v and writing the output in the
  `(b, l, h*d)` layout through the index maps (no `merge_heads` copy, no v layout copy).
- `dots` / `lib` / `free` at head_dim 64 and 256 with the flops held fixed: the mxu-fill test.
- `peaked ...`: `lib`, `free_pre` and `round_pre` again on sharper logits (q/k gain 2.6), where the
  rounding of the dominant weight shows.
"""

import json
import pathlib as pl
import subprocess

ROOT = pl.Path("/content/flashbench")

JOB = r'''
import functools, importlib.metadata as md, json, math, pathlib, statistics, sys, time
import jax, jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.experimental.pallas.ops.tpu.flash_attention as flash

OUT = pathlib.Path(sys.argv[1])
CHAIN, REPS, L = 10, 15, 4608
TRANS_B = (((1,), (1,)), ((), ()))
report = {"versions": {p: md.version(p) for p in ("jax", "jaxlib", "libtpu")},
          "device": jax.devices()[0].device_kind, "chain": CHAIN, "reps": REPS, "arms": {}}


def save(name, row):
    report["arms"][name] = row
    OUT.write_text(json.dumps(report, indent=1))
    print(name, json.dumps(row), flush=True)


def lanes(x, width):
    """a (rows, 128) per-row statistic broadcast to (rows, width), the way the library does it."""
    return x[:, :width] if width < 128 else jnp.tile(x, (1, width // 128))


def kernel(mode, block_q, block_k, scale, merged, q, k, v):
    """one attention call. grid (batch, head, q tile, k tile); the k axis accumulates in vmem scratch."""
    batch, heads, length, dim = q.shape
    steps = length // block_k

    def body(q_ref, k_ref, v_ref, o_ref, m_scr, l_scr, acc_scr):
        step = pl.program_id(3)

        @pl.when(step == 0)
        def _():
            m_scr[...] = jnp.full(m_scr.shape, -jnp.inf, jnp.float32)
            l_scr[...] = jnp.zeros(l_scr.shape, jnp.float32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, jnp.float32)

        tile_v = v_ref[0] if merged else v_ref[0, 0]
        s = lax.dot_general(q_ref[0, 0], k_ref[0, 0], TRANS_B, preferred_element_type=jnp.float32)
        if scale != 1.0:
            s = s * scale
        if mode == "dots":
            acc_scr[...] += lax.dot(s.astype(tile_v.dtype), tile_v, preferred_element_type=jnp.float32)
        elif mode == "free":
            p = jnp.exp(s)
            l_scr[...] += jnp.sum(p, axis=1)[:, None]
            acc_scr[...] += lax.dot(p.astype(tile_v.dtype), tile_v, preferred_element_type=jnp.float32)
        elif mode == "round":
            p = jnp.exp(s).astype(tile_v.dtype)
            l_scr[...] += jnp.sum(p.astype(jnp.float32), axis=1)[:, None]
            acc_scr[...] += lax.dot(p, tile_v, preferred_element_type=jnp.float32)
        else:  # "online": the library's body, line for line
            m_prev, l_prev = m_scr[...], l_scr[...]
            m_next = jnp.maximum(m_prev, jnp.max(s, axis=1)[:, None])
            p = jnp.exp(s - jnp.tile(m_next, (1, block_k // 128)))
            l_corr = jnp.exp(m_prev - m_next) * l_prev
            l_next = jnp.sum(p, axis=1)[:, None] + l_corr
            l_scr[...], m_scr[...] = l_next, m_next
            inv = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scr[...] *= lanes(l_corr * inv, dim)
            acc_scr[...] += lax.dot(p.astype(tile_v.dtype), tile_v,
                                    preferred_element_type=jnp.float32) * lanes(inv, dim)

        @pl.when(step == steps - 1)
        def _():
            out = acc_scr[...] / lanes(l_scr[...], dim) if mode in ("free", "round") else acc_scr[...]
            if merged:
                o_ref[0] = out.astype(o_ref.dtype)
            else:
                o_ref[0, 0] = out.astype(o_ref.dtype)

    q_spec = pl.BlockSpec((1, 1, block_q, dim), lambda b, h, i, j: (b, h, i, 0))
    k_spec = pl.BlockSpec((1, 1, block_k, dim), lambda b, h, i, j: (b, h, j, 0))
    if merged:  # head h is column block h of the (b, l, h*d) layout: the dma does the head transpose
        v_spec = pl.BlockSpec((1, block_k, dim), lambda b, h, i, j: (b, j, h))
        o_spec = pl.BlockSpec((1, block_q, dim), lambda b, h, i, j: (b, i, h))
        o_shape = jax.ShapeDtypeStruct((batch, length, heads * dim), q.dtype)
    else:
        v_spec, o_spec, o_shape = k_spec, q_spec, jax.ShapeDtypeStruct(q.shape, q.dtype)
    return pl.pallas_call(
        body, out_shape=o_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=(batch, heads, length // block_q, steps),
            in_specs=[q_spec, k_spec, v_spec], out_specs=o_spec,
            scratch_shapes=[pltpu.VMEM((block_q, 128), jnp.float32), pltpu.VMEM((block_q, 128), jnp.float32),
                            pltpu.VMEM((block_q, dim), jnp.float32)]),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
        name=f"attn_{mode}")(q, k, v)


def lib(block_q, block_k, scale, q, k, v):
    return flash.flash_attention(q, k, v, causal=False, sm_scale=scale, block_sizes=flash.BlockSizes(
        block_q=block_q, block_k_major=block_k, block_k=block_k, block_b=1))


def merge(x):
    b, h, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)


def inputs(heads, dim, gain=1.5):
    """q, q with 1/sqrt(d) folded in before its one bf16 rounding (what a rope table can do), k, v."""
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    def normed(key):
        x = jax.random.normal(key, (1, heads, L, dim), jnp.float32)
        return gain * x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True))
    qf = normed(kq)
    return (qf.astype(jnp.bfloat16), (qf / math.sqrt(dim)).astype(jnp.bfloat16),
            normed(kk).astype(jnp.bfloat16), jax.random.normal(kv, (1, heads, L, dim), jnp.bfloat16))


@jax.jit
def reference(q, k, v, scale):
    """fp32 softmax(scale * q k^t) v on the same bf16 inputs the arm sees."""
    qf, kf, vf = (a.astype(jnp.float32) for a in (q, k, v))
    w = jax.nn.softmax(jnp.einsum("bhld,bhsd->bhls", qf, kf, precision="highest") * scale, -1)
    return jnp.einsum("bhls,bhsd->bhld", w, vf, precision="highest")


def snr(out, ref):
    out, ref = out.astype(jnp.float32), ref.astype(jnp.float32)
    return float(10 * jnp.log10(jnp.sum(ref * ref) / jnp.sum((out - ref) ** 2)))


def run(name, fn, q, k, v, ref=None, v_merged=False):
    """compile, check one call against the reference, then time CHAIN chained calls."""
    try:
        v0 = merge(v) if v_merged else v
        one = jax.jit(fn)
        out = one(q, k, v0)
        row = {"snr_db": None if ref is None else snr(out, merge(ref) if v_merged else ref)}
        chain = jax.jit(lambda q, k, v: lax.fori_loop(0, CHAIN, lambda i, o: fn(q, k, o), v))
        for _ in range(3):
            jax.block_until_ready(chain(q, k, v0))
        ms = []
        for _ in range(REPS):
            t = time.perf_counter()
            jax.block_until_ready(chain(q, k, v0))
            ms.append(1e3 * (time.perf_counter() - t) / CHAIN)
        p50 = statistics.median(ms)
        heads, dim = q.shape[1], q.shape[3]
        row |= {"ms": round(p50, 4), "min_ms": round(min(ms), 4),
                "tflops": round(4 * heads * L * L * dim / (p50 / 1e3) / 1e12, 1)}
    except Exception as error:
        row = {"error": repr(error)[:600]}
    save(name, row)


q, qs, k, v = inputs(24, 128)
scale = 1.0 / math.sqrt(128)
ref, ref_pre = reference(q, k, v, scale), reference(qs, k, v, 1.0)
ours = lambda mode, bq, bk, sc, merged=False: functools.partial(kernel, mode, bq, bk, sc, merged)

run("lib", functools.partial(lib, 2304, 1536, scale), q, k, v, ref)
run("online", ours("online", 2304, 1536, scale), q, k, v, ref)
run("dots", ours("dots", 2304, 1536, 1.0), q, k, v)
run("free", ours("free", 2304, 1536, scale), q, k, v, ref)
run("free_pre", ours("free", 2304, 1536, 1.0), qs, k, v, ref_pre)
run("lib_pre", functools.partial(lib, 2304, 1536, 1.0), qs, k, v, ref_pre)
run("round_pre", ours("round", 2304, 1536, 1.0), qs, k, v, ref_pre)
run("round_pre q2304 k2304", ours("round", 2304, 2304, 1.0), qs, k, v, ref_pre)
run("free q2304 k2304", ours("free", 2304, 2304, scale), q, k, v, ref)
run("merged", ours("free", 2304, 1536, 1.0, True), qs, k, v, ref_pre, v_merged=True)
for bq, bk in ((4608, 1536), (4608, 768), (2304, 2304), (2304, 768), (1536, 1536), (1152, 1536),
               (1152, 1152), (1152, 768), (768, 768), (512, 512), (512, 1536), (256, 2304)):
    run(f"free_pre q{bq} k{bk}", ours("free", bq, bk, 1.0), qs, k, v, ref_pre)
    run(f"dots q{bq} k{bk}", ours("dots", bq, bk, 1.0), qs, k, v)

# same flops, different head_dim: does a 128-wide contraction starve the mxu?
for heads, dim in ((48, 64), (12, 256)):
    q2, _, k2, v2 = inputs(heads, dim)
    s2 = 1.0 / math.sqrt(dim)
    r2 = reference(q2, k2, v2, s2)
    run(f"lib d{dim}", functools.partial(lib, 2304, 1536, s2), q2, k2, v2, r2)
    run(f"dots d{dim}", ours("dots", 2304, 1536, 1.0), q2, k2, v2)
    run(f"free d{dim}", ours("free", 2304, 1536, s2), q2, k2, v2, r2)

qp, qps, kp, vp = inputs(24, 128, 2.6)
run("peaked lib", functools.partial(lib, 2304, 1536, scale), qp, kp, vp, reference(qp, kp, vp, scale))
for mode in ("free", "round"):
    run(f"peaked {mode}_pre", ours(mode, 2304, 2304, 1.0), qps, kp, vp, reference(qps, kp, vp, 1.0))

# the plain xla matmul at the two attention shapes, per head, no softmax and no 2 gb buffer kept:
# what xla's own gemm does with a 128-wide contraction (q k^t) and a 128-wide output (p v).
def xla_dots(q, k, v):
    s = jnp.einsum("bhld,bhsd->bhls", q[:, :2], k[:, :2], preferred_element_type=jnp.float32)
    return v.at[:, :2].set(jnp.einsum("bhls,bhsd->bhld", s.astype(v.dtype), v[:, :2],
                                      preferred_element_type=jnp.float32).astype(v.dtype))
run("xla_dots 2 heads (x12 for a call)", xla_dots, q, k, v)
'''

CHAIN = f"""
set -u
cd {ROOT}
pip install -q libtpu==0.0.23
echo "install exit $?" >> status
python job.py {ROOT}/report-0.0.23.json > logs/job-0.0.23.log 2>&1; echo "job 0.0.23 exit $?" >> status
echo done >> status
"""


def start() -> None:
    """write the job and chain, launch the chain detached from the kernel."""
    (ROOT / "logs").mkdir(parents=True)
    (ROOT / "job.py").write_text(JOB)
    (ROOT / "chain.sh").write_text(CHAIN)
    subprocess.Popen(["bash", str(ROOT / "chain.sh")], stdout=(ROOT / "logs/chain.log").open("w"),
                     stderr=subprocess.STDOUT, start_new_session=True)


def status() -> dict[str, object]:
    """chain markers, the report so far, and the tail of the job log."""
    read = lambda p: p.read_text(errors="replace").splitlines() if p.exists() else []  # noqa: E731
    reports = {p.name: json.loads(p.read_text()) for p in sorted(ROOT.glob("report-*.json"))}
    return {"status": read(ROOT / "status"), "reports": reports,
            "log_tail": {p.name: read(p)[-4:] for p in sorted((ROOT / "logs").glob("job-*.log"))}}


if not ROOT.exists():
    start()
print("FLASHBENCH " + json.dumps(status()))
