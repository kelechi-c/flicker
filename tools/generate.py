"""prompts -> pngs, and an optional contact sheet.

    uv run python tools/generate.py "a misty forest at dawn" --out out/images
    uv run python tools/generate.py --prompts prompts.txt --grid 5 --out out/images

weights come from `black-forest-labs/FLUX.2-klein-4B` (gated: needs `HF_TOKEN`) unless `--weights`
points at a local copy. needs a 32 GB accelerator: everything is loaded once and kept.
"""

import argparse
import json
import pathlib as pl

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from flicker.config import DIT_FILE, REPO, Config
from flicker.pipeline import Paths, generate_resident, load_all

FILES = (DIT_FILE, "text_encoder/*", "tokenizer/*", "vae/*")


def weights(root: str | None = None) -> Paths:
    """the three checkpoint paths, downloaded on first use."""
    snap = pl.Path(root or snapshot_download(REPO, allow_patterns=list(FILES)))
    return Paths(dit=str(snap / DIT_FILE),
                 text_index=str(snap / "text_encoder/model.safetensors.index.json"),
                 vae=str(snap / "vae/diffusion_pytorch_model.safetensors"))


def grid(images: list[np.ndarray], columns: int) -> np.ndarray:
    """equal-size `(h, w, 3)` images -> one sheet, row-major, `columns` wide."""
    assert len(images) % columns == 0, f"{len(images)} images do not fill rows of {columns}"
    rows = [np.concatenate(images[i:i + columns], axis=1) for i in range(0, len(images), columns)]
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--prompts", type=pl.Path, help="a file with one prompt per line")
    parser.add_argument("--out", type=pl.Path, default=pl.Path("out/images"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=0, help="columns of a contact sheet (0: none)")
    parser.add_argument("--weights", help="local snapshot of the model repo")
    args = parser.parse_args()

    prompts = args.prompt + (args.prompts.read_text().splitlines() if args.prompts else [])
    prompts = [p.strip() for p in prompts if p.strip()]
    cfg = Config()
    resident = load_all(weights(args.weights), cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    images = []
    for index, prompt in enumerate(prompts):
        images.append(generate_resident(resident, cfg, prompt, args.size, args.size, args.seed))
        Image.fromarray(images[-1]).save(args.out / f"{index:02d}.png")
        print(f"{index:02d} {prompt[:80]}", flush=True)
    (args.out / "prompts.json").write_text(json.dumps(
        {"size": args.size, "seed": args.seed, "prompts": prompts}, indent=1))
    if args.grid:
        Image.fromarray(grid(images, args.grid)).save(args.out / "grid.png")


if __name__ == "__main__":
    main()
