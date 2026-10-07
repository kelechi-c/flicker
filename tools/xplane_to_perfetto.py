"""Convert XProf `.xplane.pb` traces into Perfetto-viewable `.json.gz`.

    uv run --no-project --with xprof python tools/xplane_to_perfetto.py out/trace/trace

`jax.profiler.trace(..., create_perfetto_trace=True)` writes two things per region: a
`perfetto_trace.json.gz` (a Chrome-trace JSON, already Perfetto-shaped) and one or more
`*.xplane.pb` (XProf's own format, which carries more: per-op HLO metadata, the full op hierarchy
rather than the flattened view, and the counters XProf tracks). The xplane is typically 2-4x the
size of the Perfetto JSON for the same region.

Converting is what Sail Research describe for their TPU v6e work: open the `.xplane.pb` in XProf, or
convert it with `xprof.convert.raw_to_tool_data.xspace_to_tool_data(paths, "trace_viewer@", {})` and
load the result in Perfetto. This is the gist they link
(https://gist.github.com/transmissions11/0de193188187a1a590a669fbcde26240), factored into a CLI.

Output naming: `<region>.xplane.json.gz` next to the source, and `<region>` is the first path
component under the root, which is how `tools/trace_shots.py` finds regions.
"""

import argparse
import gzip
import pathlib as pl
import sys
from typing import Any


def convert(xplane: pl.Path, target: pl.Path) -> tuple[int, int]:
    """Write `xplane` as a gzipped Chrome-trace JSON; return (input bytes, output bytes)."""
    from xprof.convert import raw_to_tool_data

    data, _ = raw_to_tool_data.xspace_to_tool_data([str(xplane)], "trace_viewer@", {})
    if isinstance(data, str):
        data = data.encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(data))
    return xplane.stat().st_size, target.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pl.Path, help="dir holding <region>/plugins/profile/<stamp>/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    xplanes = sorted(args.root.rglob("*.xplane.pb"))
    if not xplanes:
        print(f"no .xplane.pb under {args.root}", file=sys.stderr)
        return 1
    for xplane in xplanes:
        region = xplane.relative_to(args.root).parts[0]
        target = xplane.parent / f"{region}.xplane.json.gz"  # beside plugins/profile/<stamp>/
        if target.exists() and not args.force:
            print(f"  {region:16s} skip (exists)")
            continue
        raw, out = convert(xplane, target)
        print(f"  {region:16s} {raw / 2**20:6.2f} MiB xplane -> {out / 2**20:6.2f} MiB perfetto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
