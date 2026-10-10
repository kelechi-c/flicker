from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Sequence
from typing import Any, Protocol

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
LUMA_MEAN_BOUNDS = (0.05, 0.95)
LUMA_STD_FLOOR = 0.02
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


class QualityError(Exception):
    pass


class Scorer(Protocol):

    name: str

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float: ...

    def meta(self) -> Json: ...


def decode_rgb(path: Any, *, size: tuple[int, int] | None = None) -> Array:
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
    return np.asarray(rgb, dtype=np.float32) / 255.0


def to_nchw(rgb: Array) -> Array:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise QualityError(f"expected (H, W, 3) RGB, got {array.shape}")
    return rgb_float(array).transpose(2, 0, 1)[None]


def rgb_stats(rgb: Array) -> Json:
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
    if np.asarray(ref_rgb).shape != np.asarray(test_rgb).shape:
        raise QualityError(f"pair shapes differ: {np.asarray(ref_rgb).shape} vs "
                           f"{np.asarray(test_rgb).shape}")


def package_version(name: str) -> str | None:
    return importlib.metadata.version(name) if importlib.util.find_spec(name) else None


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def scorer_versions() -> Json:
    return {"pyiqa": package_version("pyiqa"), "torch": package_version("torch"),
            "transformers": package_version("transformers"), "numpy": np.__version__,
            "pyiqa_min_version": PYIQA_MIN_VERSION, "lpips_metric": LPIPS_METRIC,
            "lpips_net": LPIPS_NET, "lpips_version": LPIPS_VERSION,
            "dinov2_model": DINOV2_MODEL, "dinov2_revision": DINOV2_REVISION}


class LpipsScorer:

    name = "lpips"

    def __init__(self, *, device: str = LPIPS_DEVICE, net: str = LPIPS_NET,
                 version: str = LPIPS_VERSION, **opts: Any) -> None:
        if importlib.util.find_spec("pyiqa") is None:
            raise QualityError(f"pyiqa is not installed; install the `quality` extra "
                               f"(pyiqa=={PYIQA_MIN_VERSION}) to score LPIPS")
        import pyiqa

        installed = package_version("pyiqa")
        if installed is None or _version_tuple(installed) < _version_tuple(PYIQA_MIN_VERSION):
            raise QualityError(f"pyiqa {installed} < pinned {PYIQA_MIN_VERSION}")
        self._device, self._net, self._version = device, net, version
        self._opts = dict(opts)
        self._metric = pyiqa.create_metric(LPIPS_METRIC, device=device, net=net, version=version,
                                           **opts)

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float:
        import torch
        _require_same_shape(ref_rgb, test_rgb)
        left, right = to_nchw(ref_rgb), to_nchw(test_rgb)
        with torch.no_grad():
            score = self._metric(torch.from_numpy(left), torch.from_numpy(right))
        return float(np.asarray(score.detach().cpu(), dtype=np.float64).reshape(-1).mean())

    def meta(self) -> Json:
        return {"name": self.name, "metric": LPIPS_METRIC, "net": self._net,
                "version": self._version, "device": self._device, "opts": self._opts,
                "lower_better": bool(getattr(self._metric, "lower_better", True)),
                "input": "native 1024² RGB float32 [0,1] NCHW", **scorer_versions()}


class DinoScorer:

    name = "dino_cls_cosine"

    def __init__(self, *, device: str = "cpu", model_id: str = DINOV2_MODEL,
                 revision: str = DINOV2_REVISION) -> None:
        if importlib.util.find_spec("transformers") is None or \
                importlib.util.find_spec("torch") is None:
            raise QualityError("transformers and torch are required for the DINOv2 gate; install "
                               "the `quality` extra")
        from transformers import AutoImageProcessor, AutoModel

        self._device, self._model_id, self._revision = device, model_id, revision
        self._processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()

    def _cls_features(self, images: Sequence[Array]) -> Any:
        import torch
        batch = self._processor(images=[Image.fromarray(np.asarray(image)) for image in images],
                                return_tensors="pt").to(self._device)
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        return hidden[:, 0, :]

    def score_pair(self, ref_rgb: Array, test_rgb: Array) -> float:
        import torch
        _require_same_shape(ref_rgb, test_rgb)
        features = torch.nn.functional.normalize(
            self._cls_features([ref_rgb, test_rgb]), dim=-1)
        return float((features[0] * features[1]).sum().item())

    def meta(self) -> Json:
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
    built: list[Scorer] = []
    for name in names:
        if name == "lpips":
            built.append(LpipsScorer())
        elif name in {"dino", "dino_cls_cosine"}:
            built.append(DinoScorer())
        else:
            raise QualityError(f"unknown scorer {name!r}; use lpips, dino or none")
    return built
