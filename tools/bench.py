import argparse
import json
import pathlib as pl
import statistics
import time
from collections.abc import Callable

import jax
from generate import weights

from flicker.config import Config
from flicker.pipeline import (
    decode_from,
    denoise_from,
    encode_ctx,
    generate_resident,
    load_all,
)

PROMPT = "a car jumping off of a cliff with a crowd cheering"
WARMUPS = 3
FLOPS_1024 = 153.0e12


def p50_ms(fn: Callable[[], object], reps: int) -> float:
    for _ in range(WARMUPS):
        jax.block_until_ready(fn())
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(1e3 * (time.perf_counter() - start))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--weights", help="local snapshot of the model repo")
    parser.add_argument("--trace", type=pl.Path, help="write an xprof trace of 3 requests here")
    args = parser.parse_args()

    cfg, size = Config(), args.size
    start = time.perf_counter()
    res = load_all(weights(args.weights), cfg)
    load_s = time.perf_counter() - start
    key = jax.random.key(0)
    request = lambda: generate_resident(res, cfg, PROMPT, size, size)

    start = time.perf_counter()
    request()
    cold_s = time.perf_counter() - start
    ctx = encode_ctx(res.text, cfg, PROMPT)
    latent = denoise_from(res.dit, cfg, ctx, size, size, key)
    report = {
        "device": jax.devices()[0].device_kind, "jax": jax.__version__, "size": size, "reps": args.reps,
        "load_s": round(load_s, 1), "first_request_s": round(cold_s, 1),
        "request_ms": p50_ms(request, args.reps),
        "text_ms": p50_ms(lambda: encode_ctx(res.text, cfg, PROMPT), args.reps),
        "dit_ms": p50_ms(lambda: denoise_from(res.dit, cfg, ctx, size, size, key), args.reps),
        "vae_ms": p50_ms(lambda: decode_from(res.vae, cfg, latent, size, size), args.reps),
    }
    report["img_per_s"] = 1e3 / report["request_ms"]
    if size == 1024:
        report["tflops_per_s"] = FLOPS_1024 / report["request_ms"] / 1e9
    if args.trace:
        with jax.profiler.trace(str(args.trace)):
            for _ in range(3):
                request()
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
