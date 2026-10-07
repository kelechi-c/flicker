"""Heavyweight image-quality scorers for the 1024² freeze: LPIPS (pyiqa, `net=alex`) and DINOv2.

Two primary B.6 gates live here and nothing else does:

* **LPIPS** — `pyiqa.create_metric('lpips', device='cpu', net='alex')` (pinned `pyiqa==0.1.16`,
  `version=0.1`, `lower_better=True`), fed RGB float32 `[0,1]` NCHW at the native 1024², so the
  only resampling is whatever pyiqa's own arch does.
* **DINOv2 CLS cosine** — pinned `facebook/dinov2-base` at revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415` through the canonical pinned `AutoImageProcessor`,
  `last_hidden_state[:, 0, :]`, L2-normalized, cosine of the two CLS features.

Both read the **same decoded RGB original**: `decode_rgb` is the only place a PNG becomes pixels, and
DINO applies its own documented processor resize/normalization on top. torch / pyiqa / transformers
are imported lazily inside the constructors, so this module — and any test injecting fake scorers —
imports without them and the JAX serving path never pays for them.

Scoring is deliberately **off TPU**: these are CPU/GPU host work (`docs/benchmarks.md` B.9).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

Json = dict[str, Any]
Array = np.ndarray

SIZE = 1024
LPIPS_METRIC = "lpips"
LPIPS_NET = "alex"
LPIPS_VERSION = "0.1"
LPIPS_DEVICE = "cpu"
PYIQA_MIN_VERSION = "0.1.16"
DINOV2_MODEL = "facebook/dinov2-base"
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINO_POOLING = "last_hidden_state[:, 0, :] (CLS), l2-normalized"
# grey-blob hard-fail bounds from B.6: mean luminance outside [0.05, 0.95], or std < 0.02
LUMA_MEAN_BOUNDS = (0.05, 0.95)
LUMA_STD_FLOOR = 0.02
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


class QualityError(Exception):
    """A scorer, decoder or dependency contract was violated."""


class Scorer(Protocol):
    """What the harness needs from a scorer; both real scorers and test fakes implement this."""

    name: str

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float: ...

    def meta(self) -> Json: ...


# ---------------------------------------------------------------------------- pixels
def decode_rgb(path: Any, *, size: tuple[int, int] | None = None) -> Array:
    """Decode a PNG to `(H, W, 3)` uint8 RGB — the single decode both gates read.

    Non-RGB modes (grayscale, palette) are converted to RGB rather than rejected, because the
    comparison is RGB end to end; a wrong resolution *is* rejected when `size` is given, since a
    downscaled pair silently changes LPIPS without any error.
    """
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except FileNotFoundError as exc:
        raise QualityError(f"missing image: {path}") from exc
    except OSError as exc:
        raise QualityError(f"undecodable image {path}: {exc}") from exc
    if size is not None and rgb.shape[:2] != tuple(size):
        raise QualityError(f"{path}: expected {tuple(size)} RGB, got {rgb.shape[:2]}")
    return rgb


def rgb_float(rgb: Array) -> Array:
    """uint8 RGB to float32 `[0,1]`; the LPIPS input convention."""
    return np.asarray(rgb, dtype=np.float32) / 255.0


def to_nchw(rgb: Array) -> Array:
    """`(H, W, 3)` uint8 RGB to `(1, 3, H, W)` float32 `[0,1]` — the NCHW convention pyiqa wants."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise QualityError(f"expected (H, W, 3) RGB, got {array.shape}")
    return rgb_float(array).transpose(2, 0, 1)[None]


def rgb_stats(rgb: Array) -> Json:
    """Luminance/grey-blob diagnostics for B.6's NaN and collapsed-output hard-fail gate.

    `hard_fail` is true for non-finite pixels, a mean luminance outside `[0.05, 0.95]` or a
    luminance std below `0.02` (a grey blob), with the triggering reason kept as text.
    """
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise QualityError(f"expected (H, W, 3) RGB, got {array.shape}")
    values = np.asarray(array, dtype=np.float64) / 255.0
    finite_pixels = bool(np.isfinite(values).all())
    luminance = values @ _LUMA if values.size else np.zeros((0,), dtype=np.float64)
    mean = float(luminance.mean()) if luminance.size else None
    std = float(luminance.std()) if luminance.size else None
    reasons = []
    if not finite_pixels:
        reasons.append("non-finite pixel values")
    if mean is None or not LUMA_MEAN_BOUNDS[0] <= mean <= LUMA_MEAN_BOUNDS[1]:
        reasons.append(f"mean luminance {mean} outside {LUMA_MEAN_BOUNDS}")
    if std is None or std < LUMA_STD_FLOOR:
        reasons.append(f"luminance std {std} below {LUMA_STD_FLOOR} (grey blob)")
    return {"mean_luminance": mean, "std_luminance": std, "nonfinite": not finite_pixels,
            "grey_blob": std is not None and std < LUMA_STD_FLOOR, "hard_fail": bool(reasons),
            "reason": "; ".join(reasons) if reasons else None}


def _require_same_shape(ref_rgb: Array, test_rgb: Array) -> None:
    """Both gates are paired at one resolution; a size difference is an error, not a score."""
    if np.asarray(ref_rgb).shape != np.asarray(test_rgb).shape:
        raise QualityError(f"pair shapes differ: {np.asarray(ref_rgb).shape} vs "
                           f"{np.asarray(test_rgb).shape}")


# ---------------------------------------------------------------------------- versions
def package_version(name: str) -> str | None:
    """Installed distribution version, or `None` when the metrics environment lacks it."""
    return importlib.metadata.version(name) if importlib.util.find_spec(name) else None


def _version_tuple(text: str) -> tuple[int, ...]:
    """`'0.1.16'` -> `(0, 1, 16)`; a non-numeric suffix is ignored for the minimum check."""
    parts = []
    for piece in text.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def scorer_versions() -> Json:
    """Every version a score depends on, `null` where the metrics environment lacks it."""
    return {"pyiqa": package_version("pyiqa"), "torch": package_version("torch"),
            "transformers": package_version("transformers"), "numpy": np.__version__,
            "pyiqa_min_version": PYIQA_MIN_VERSION, "lpips_metric": LPIPS_METRIC,
            "lpips_net": LPIPS_NET, "lpips_version": LPIPS_VERSION,
            "dinov2_model": DINOV2_MODEL, "dinov2_revision": DINOV2_REVISION}


# ---------------------------------------------------------------------------- LPIPS
class LpipsScorer:
    """`pyiqa` LPIPS on `net='alex'`; lower is better, identical images should score ~0.

    Construction is the fail-closed dependency check: an absent pyiqa, a version below
    `PYIQA_MIN_VERSION` (0.1.16) or an absent net is a `QualityError`, never a silent fallback to a
    different scorer. `opts` is forwarded to `create_metric` for an explicit non-default choice and
    recorded verbatim in `meta()`.
    """

    name = "lpips"

    def __init__(self, *, device: str = LPIPS_DEVICE, net: str = LPIPS_NET,
                 version: str = LPIPS_VERSION, **opts: Any) -> None:
        if importlib.util.find_spec("pyiqa") is None:
            raise QualityError(f"pyiqa is not installed; install the `quality` extra "
                               f"(pyiqa=={PYIQA_MIN_VERSION}) to score LPIPS")
        import pyiqa  # noqa: PLC0415 — lazy: torch stays out of the JAX serving path

        installed = package_version("pyiqa")
        if installed is None or _version_tuple(installed) < _version_tuple(PYIQA_MIN_VERSION):
            raise QualityError(f"pyiqa {installed} < pinned {PYIQA_MIN_VERSION}")
        self._device, self._net, self._version = device, net, version
        self._opts = dict(opts)
        self._metric = pyiqa.create_metric(LPIPS_METRIC, device=device, net=net, version=version,
                                           **opts)

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float:
        """Paired LPIPS at native resolution; both inputs are `(H, W, 3)` uint8 RGB."""
        import torch  # noqa: PLC0415
        _require_same_shape(ref_rgb, test_rgb)
        left, right = to_nchw(ref_rgb), to_nchw(test_rgb)
        with torch.no_grad():
            score = self._metric(torch.from_numpy(left), torch.from_numpy(right))
        return float(np.asarray(score.detach().cpu(), dtype=np.float64).reshape(-1).mean())

    def meta(self) -> Json:
        """Everything needed to reproduce or distrust the score: net, version, options, resolved
        `lower_better`, device and package versions."""
        return {"name": self.name, "metric": LPIPS_METRIC, "net": self._net,
                "version": self._version, "device": self._device, "opts": self._opts,
                "lower_better": bool(getattr(self._metric, "lower_better", True)),
                "input": "native 1024² RGB float32 [0,1] NCHW", **scorer_versions()}


# ---------------------------------------------------------------------------- DINOv2
class DinoScorer:
    """Pinned `facebook/dinov2-base` CLS-feature cosine; higher is better, identical images ~1.

    The canonical `AutoImageProcessor` is used unchanged so its resize/crop/normalization are the
    published ones; features are `last_hidden_state[:, 0, :]`, L2-normalized, and the score is the
    cosine of the pair. The revision is pinned to the model id's commit rather than `main`.
    """

    name = "dino_cls_cosine"

    def __init__(self, *, device: str = "cpu", model_id: str = DINOV2_MODEL,
                 revision: str = DINOV2_REVISION) -> None:
        if importlib.util.find_spec("transformers") is None or \
                importlib.util.find_spec("torch") is None:
            raise QualityError("transformers and torch are required for the DINOv2 gate; install "
                               "the `quality` extra")
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        self._device, self._model_id, self._revision = device, model_id, revision
        self._processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()

    def _cls_features(self, images: Sequence[Array]) -> Any:
        """CLS features for decoded RGB images, under the processor's own preprocessing."""
        import torch  # noqa: PLC0415
        batch = self._processor(images=[Image.fromarray(np.asarray(image)) for image in images],
                                return_tensors="pt").to(self._device)
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        return hidden[:, 0, :]

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float:
        """Cosine of the two L2-normalized CLS features; identical images score 1."""
        import torch  # noqa: PLC0415
        _require_same_shape(ref_rgb, test_rgb)
        features = torch.nn.functional.normalize(
            self._cls_features([ref_rgb, test_rgb]), dim=-1)
        return float((features[0] * features[1]).sum().item())

    def meta(self) -> Json:
        """The pinned model, pooling and processor preprocessing actually resolved."""
        processor = self._processor
        return {"name": self.name, "model_id": self._model_id, "revision": self._revision,
                "pooling": DINO_POOLING, "device": self._device,
                "processor": {"class": type(processor).__name__,
                              "size": getattr(processor, "size", None),
                              "crop_size": getattr(processor, "crop_size", None),
                              "image_mean": getattr(processor, "image_mean", None),
                              "image_std": getattr(processor, "image_std", None),
                              "resample": getattr(processor, "resample", None)},
                **scorer_versions()}


def default_scorers(names: Sequence[str] = ("lpips", "dino")) -> list[Scorer]:
    """Construct the requested primary scorers; an unknown name is a `QualityError`.

    Construction happens once per run, before any image is decoded, so a missing metrics
    environment fails immediately instead of after 96 pairs.
    """
    built: list[Scorer] = []
    for name in names:
        if name == "lpips":
            built.append(LpipsScorer())
        elif name in {"dino", "dino_cls_cosine"}:
            built.append(DinoScorer())
        else:
            raise QualityError(f"unknown scorer {name!r}; use lpips, dino or none")
    return built
