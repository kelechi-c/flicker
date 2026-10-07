"""per-stage and per-op device time of a traced request, from an xprof trace. local, no accelerator.

    uv run --no-project --with xprof python tools/op_profile.py out/trace --iterations 3

xprof's `op_profile` gives every hlo op an inclusive time, flop count and bytes accessed; subtracting
the children leaves each op's own share, and those shares sum back to the device total. ops are then
grouped by the stage whose jit they sit in and by the python line that produced them, so the table
needs no knowledge of this revision's line numbers. `--iterations` is how many requests the trace
holds (3 for `tools/bench.py --trace`); every figure is per request.
"""

import argparse
import collections
import json
import pathlib as pl
from typing import Any, NamedTuple

Json = dict[str, Any]
STAGES = ("jit(encode)", "jit(denoise)", "jit(vae_decode)")


class Op(NamedTuple):
    """one hlo op's own (exclusive) share of the trace."""

    stage: str
    source: str
    jax_op: str
    seconds: float
    flops: float
    bytes: float


def tree(xplane: pl.Path) -> Json:
    """xplane -> xprof's hierarchical op profile."""
    from xprof.convert import raw_to_tool_data

    data, ok = raw_to_tool_data.xspace_to_tool_data([str(xplane)], "op_profile", {})
    if not ok:
        raise RuntimeError(f"xprof refused op_profile for {xplane}")
    return json.loads(data.decode() if isinstance(data, bytes) else data)["byProgram"]


def flatten(node: Json) -> list[Op]:
    """every node as an `Op`, with its children's metrics subtracted."""
    children = node.get("children") or []
    metrics, info = node["metrics"], node.get("xla") or {}
    own = lambda key: metrics[key] - sum(c["metrics"][key] for c in children)  # noqa: E731
    moved = sum(metrics["rawBytesAccessedArray"][i] - sum(c["metrics"]["rawBytesAccessedArray"][i]
                                                         for c in children) for i in (0, 2))
    prov = (info.get("provenance") or "").rstrip(":")
    source = info.get("sourceInfo") or {}
    stage = "idle" if node["name"] == "IDLE" else next((s for s in STAGES if s in prov), "other")
    line = f"{(source.get('fileName') or '').rsplit('/', 1)[-1]}:{source.get('lineNumber') or 0}"
    op = Op(stage, line, prov.rsplit("/", 1)[-1], own("rawTime") / 1e12, own("rawFlops"), moved)
    return [op] + [leaf for child in children for leaf in flatten(child)]


def table(rows: dict[tuple[str, ...], list[float]], iterations: int, limit: int) -> list[str]:
    """`ms  tflop/s  gb/s  key` lines, largest first."""
    out = []
    for key, (seconds, flops, moved) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:limit]:
        rate = lambda x: x / seconds / 1e12 if seconds else 0.0  # noqa: E731
        out.append(f"{1e3 * seconds / iterations:9.3f} ms {rate(flops):7.1f} tf/s "
                   f"{1e3 * rate(moved):7.0f} gb/s  {'  '.join(key)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=pl.Path, help="a trace directory or one .xplane.pb")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--top", type=int, default=15, help="ops listed per stage")
    args = parser.parse_args()

    xplane = args.trace if args.trace.is_file() else max(args.trace.rglob("*.xplane.pb"),
                                                         key=lambda p: p.stat().st_mtime)
    ops = flatten(tree(xplane))
    stages: dict[tuple[str, ...], list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    lines: dict[str, dict[tuple[str, ...], list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0, 0.0]))
    for op in ops:
        for bucket in (stages[(op.stage,)], lines[op.stage][(op.source, op.jax_op)]):
            bucket[0] += op.seconds
            bucket[1] += op.flops
            bucket[2] += op.bytes
    print("\n".join(table(stages, args.iterations, len(stages))))
    for stage in STAGES:
        print(f"\n{stage}\n" + "\n".join(table(lines[stage], args.iterations, args.top)))


if __name__ == "__main__":
    main()
