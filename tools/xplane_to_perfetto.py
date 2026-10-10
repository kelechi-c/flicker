import argparse
import gzip
import pathlib as pl
import sys


def convert(xplane: pl.Path, target: pl.Path) -> tuple[int, int]:
    from xprof.convert import raw_to_tool_data

    data, _ = raw_to_tool_data.xspace_to_tool_data([str(xplane)], "trace_viewer@", {})
    if isinstance(data, str):
        data = data.encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(data))
    return xplane.stat().st_size, target.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pl.Path, help="dir holding <region>/plugins/profile/<stamp>/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    xplanes = sorted(args.root.rglob("*.xplane.pb"))
    if not xplanes:
        print(f"no .xplane.pb under {args.root}", file=sys.stderr)
        return 1
    for xplane in xplanes:
        region = xplane.relative_to(args.root).parts[0]
        target = xplane.parent / f"{region}.xplane.json.gz"
        if target.exists() and not args.force:
            print(f"  {region:16s} skip (exists)")
            continue
        raw, out = convert(xplane, target)
        print(f"  {region:16s} {raw / 2**20:6.2f} MiB xplane -> {out / 2**20:6.2f} MiB perfetto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
