"""the vae's `nearest-2x + 3x3 conv`, three ways, on one colab v6e-1. no weights, no token.

one file for `colab exec -s <session> -f tools/bench_upsample.py`: the first call starts a detached job,
every call prints a status json, re-exec to poll.

the decoder's three upsamplers at 1024² cost 2.78 tflop of convolution plus an 8.7 ms materialised
broadcast (`out/tpu/d4` profile). `nearest-2x` is zero insertion followed by a 2x2 box, so the pair is
exactly one convolution, and it can be written two ways that never build the upsampled tensor:

- `nearest`: what ships — `nn.upsample_nearest` then a 3x3 same conv at high resolution (36 c² hw flops).
- `dilated`: one 4x4 conv with `lhs_dilation=2` on the low-res input, kernel = the 3x3 summed over the box.
- `phases`: four 2x2 convs on the low-res input, one per output parity, interleaved (16 c² hw flops).

random bf16 weights at the real shapes; each arm's snr is against the fp32 `nearest` result.
"""

import json
import pathlib as pl
import subprocess

ROOT = pl.Path("/content/upbench")

JOB = r'''
import json, pathlib, statistics, sys, time
import jax, jax.numpy as jnp
from jax import lax

OUT, REPS = pathlib.Path(sys.argv[1]), 20
DN = ("NHWC", "HWIO", "NHWC")
report = {"device": jax.devices()[0].device_kind, "jax": jax.__version__, "arms": {}}


def conv(x, w, padding, dilation=(1, 1)):
    return lax.conv_general_dilated(x, w, (1, 1), padding, lhs_dilation=dilation, dimension_numbers=DN,
                                    preferred_element_type=jnp.float32)


def upsample_nearest(x):
    n, h, w, c = x.shape
    return jnp.broadcast_to(x[:, :, None, :, None, :], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)


def nearest(x, w, b):
    return (conv(upsample_nearest(x), w, "SAME").astype(x.dtype) + b)


def box_kernel(w):
    """the 3x3 kernel summed over the 2x2 box nearest-upsampling applies: (4, 4, in, out)."""
    z = jnp.zeros((4, 4, *w.shape[2:]), jnp.float32)
    return z.at[:3, :3].add(w).at[1:, :3].add(w).at[:3, 1:].add(w).at[1:, 1:].add(w)


def dilated(x, w4, b):
    return conv(x, w4, ((2, 2), (2, 2)), (2, 2)).astype(x.dtype) + b


def phase_kernels(w):
    """output pixel (2i+a, 2j+b) sees low-res rows {i-1, i} when a=0 and {i, i+1} when a=1."""
    def fold(t, parity):  # t: three taps on the leading axis -> the two that survive
        return jnp.stack((t[0], t[1] + t[2]) if parity == 0 else (t[0] + t[1], t[2]))
    return [[jnp.swapaxes(fold(jnp.swapaxes(fold(w, a), 0, 1), b), 0, 1) for b in (0, 1)] for a in (0, 1)]


def phases(x, kernels, b):
    outs = [[conv(x, kernels[a][c], ((1 - a, a), (1 - c, c))) for c in (0, 1)] for a in (0, 1)]
    n, h, w, ch = outs[0][0].shape
    full = jnp.stack([jnp.stack(row, axis=3) for row in outs], axis=2).reshape(n, 2 * h, 2 * w, ch)
    return full.astype(x.dtype) + b


def snr(out, ref):
    out, ref = out.astype(jnp.float32), ref.astype(jnp.float32)
    return float(10 * jnp.log10(jnp.sum(ref * ref) / jnp.sum((out - ref) ** 2)))


def run(name, fn, args, ref):
    try:
        f = jax.jit(fn)
        row = {"snr_db": snr(f(*args), ref)}
        for _ in range(3):
            jax.block_until_ready(f(*args))
        ms = []
        for _ in range(REPS):
            t = time.perf_counter()
            jax.block_until_ready(f(*args))
            ms.append(1e3 * (time.perf_counter() - t))
        row |= {"ms": round(statistics.median(ms), 3), "min_ms": round(min(ms), 3)}
    except Exception as error:
        row = {"error": repr(error)[:500]}
    report["arms"][name] = row
    OUT.write_text(json.dumps(report, indent=1))
    print(name, json.dumps(row), flush=True)


for side, ch in ((128, 512), (256, 512), (512, 256)):
    kx, kw, kb = jax.random.split(jax.random.key(side), 3)
    x = jax.random.normal(kx, (1, side, side, ch), jnp.bfloat16)
    wf = jax.random.normal(kw, (3, 3, ch, ch), jnp.float32) / (3 * ch ** 0.5)
    w, b = wf.astype(jnp.bfloat16), jax.random.normal(kb, (ch,), jnp.bfloat16)
    ref = jax.jit(nearest)(x.astype(jnp.float32), w.astype(jnp.float32), b.astype(jnp.float32))
    tag = f"{side}->{2 * side} c{ch}"
    run(f"nearest {tag}", nearest, (x, w, b), ref)
    # fold from the bf16 weights the checkpoint holds, in fp32, then round once
    w32 = w.astype(jnp.float32)
    run(f"dilated {tag}", dilated, (x, box_kernel(w32).astype(jnp.bfloat16), b), ref)
    run(f"phases {tag}", phases, (x, [[k.astype(jnp.bfloat16) for k in r] for r in phase_kernels(w32)], b), ref)
'''


def start() -> None:
    (ROOT / "logs").mkdir(parents=True)
    (ROOT / "job.py").write_text(JOB)
    cmd = f"cd {ROOT} && python job.py {ROOT}/report.json > logs/job.log 2>&1; echo \"job exit $?\" >> status"
    subprocess.Popen(["bash", "-c", cmd], start_new_session=True)


def status() -> dict[str, object]:
    read = lambda p: p.read_text(errors="replace").splitlines() if p.exists() else []  # noqa: E731
    report = ROOT / "report.json"
    return {"status": read(ROOT / "status"), "log_tail": read(ROOT / "logs/job.log")[-4:],
            "report": json.loads(report.read_text()) if report.exists() else None}


if not ROOT.exists():
    start()
print("UPBENCH " + json.dumps(status()))
