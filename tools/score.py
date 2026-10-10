import argparse
import json
import pathlib as pl

import numpy as np
from quality_metrics import decode_rgb, default_scorers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=pl.Path)
    parser.add_argument("candidate", type=pl.Path)
    args = parser.parse_args()

    names = sorted(p.name for p in args.reference.glob("*.png") if (args.candidate / p.name).exists())
    assert names, "no png present in both directories"
    lpips, dino = default_scorers()
    pairs = [(decode_rgb(args.reference / n), decode_rgb(args.candidate / n)) for n in names]
    scores = {s.name: [s.score_pair(ref, test) for ref, test in pairs] for s in (lpips, dino)}
    print(json.dumps({
        "pairs": len(names),
        "lpips_median": float(np.median(scores[lpips.name])),
        "dino_p5": float(np.percentile(scores[dino.name], 5)),
        "per_image": {n: {k: v[i] for k, v in scores.items()} for i, n in enumerate(names)},
    }, indent=1))


if __name__ == "__main__":
    main()
