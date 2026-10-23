"""Foreground segmentation for photos.

AI matting runs through rembg's ONNX sessions, always with an explicitly
named model: rembg's own default (BRIA RMBG) is licensed for non-commercial
use only and is never loaded here. Supported models and their licences:

  birefnet       BiRefNet general (Swin-L), MIT        best edges, ~1 GB, GPU recommended
  birefnet-lite  BiRefNet general lite (Swin-T), MIT   faster, ~220 MB
  isnet          IS-Net general use (DIS), Apache-2.0  ~180 MB
  u2net          U^2-Net, Apache-2.0                   oldest, ~170 MB
  classic        background colour + Otsu (no AI)      deterministic, best for line drawings

`auto` keeps an existing alpha channel, sends line drawings to `classic`
(AI matting is trained on photos), and otherwise tries birefnet then isnet on a
CUDA GPU, or isnet only on CPU (BiRefNet at 1024 px takes ~20 s and ~7 GB RAM
per image on a 4-core CPU), then classic. V3D_SEGMENTER names the model to try
first. The method string reports the execution provider (cuda/cpu).
"""
from __future__ import annotations

import os

import numpy as np

from ..raster.image_input import _drop_small, silhouette_mask

MODELS = {
    "birefnet": ("birefnet-general", "MIT (ZhengPeng7/BiRefNet)"),
    "birefnet-lite": ("birefnet-general-lite", "MIT (ZhengPeng7/BiRefNet)"),
    "isnet": ("isnet-general-use", "Apache-2.0 (xuebinqin/DIS)"),
    "u2net": ("u2net", "Apache-2.0 (xuebinqin/U-2-Net)"),
}
AUTO_ORDER = ("birefnet", "isnet")
CPU_AUTO_ORDER = ("isnet",)
_SESSIONS: dict = {}


def has_cuda() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def _device() -> str:
    return "cuda" if has_cuda() else "cpu"


def _session(key: str):
    """Cached rembg session; a model that failed to load is not retried in this process."""
    import onnxruntime as ort
    from rembg import new_session

    if key not in _SESSIONS:
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False   # the arena grows by GBs per call with BiRefNet on CPU
        opts.enable_mem_pattern = False
        try:
            _SESSIONS[key] = new_session(MODELS[key][0], sess_opts=opts)
        except Exception as exc:  # noqa: BLE001
            _SESSIONS[key] = exc
    if isinstance(_SESSIONS[key], Exception):
        raise RuntimeError(f"{key} unavailable: {_SESSIONS[key]}")
    return _SESSIONS[key]


def ai_alpha(rgb: np.ndarray, model: str) -> np.ndarray:
    """Soft alpha (H, W) uint8 from one AI matting model."""
    from rembg import remove

    out = remove(np.ascontiguousarray(rgb[..., :3]), session=_session(model), only_mask=True)
    return np.asarray(out, dtype=np.uint8)


def matte(rgba: np.ndarray, backend: str = "auto") -> tuple[np.ndarray, str]:
    """Soft alpha (H, W) uint8 and the method used.

    backend: auto | birefnet | birefnet-lite | isnet | u2net | classic
    ("rembg" is accepted as an alias of birefnet).
    """
    backend = {"rembg": "birefnet"}.get(backend, backend)
    if rgba[..., 3].min() < 250:                     # already cut out
        mask, method = silhouette_mask(rgba)
        return np.where(mask, rgba[..., 3], 0).astype(np.uint8), method
    if backend == "classic":
        mask, method = silhouette_mask(rgba)
        return (mask * 255).astype(np.uint8), method
    if backend == "auto":
        mask, method = silhouette_mask(rgba)
        if "line_art" in method:                     # a drawing: AI matting would cut the strokes
            return (mask * 255).astype(np.uint8), method
        preferred = os.environ.get("V3D_SEGMENTER")
        order = ((preferred,) if preferred in MODELS else ()) + (AUTO_ORDER if has_cuda() else CPU_AUTO_ORDER)
        for key in dict.fromkeys(order):
            try:
                alpha = ai_alpha(rgba, key)
                if (alpha > 127).any():
                    return alpha, f"{key} ({MODELS[key][1]}, {_device()})"
            except Exception:  # noqa: BLE001 - rembg missing, offline, or model failed: try the next one
                continue
        return (mask * 255).astype(np.uint8), method
    if backend not in MODELS:
        raise ValueError(f"unknown segmenter {backend!r}; choose auto, classic or one of {sorted(MODELS)}")
    return ai_alpha(rgba, backend), f"{backend} ({MODELS[backend][1]}, {_device()})"


def cutout(rgba: np.ndarray, backend: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Foreground mask, soft alpha and the method used, from one model call."""
    alpha, method = matte(rgba, backend)
    mask = alpha > 127
    if mask.any() and "classic" not in method and method.split(" ")[0] in MODELS:
        mask = _drop_small(mask, max(16, int(mask.sum()) // 2000))
        alpha = np.where(mask, alpha, np.minimum(alpha, 127)).astype(np.uint8)
    return mask, alpha, method


def segment(rgba: np.ndarray, backend: str = "auto") -> tuple[np.ndarray, str]:
    """Foreground mask and the method used."""
    mask, _, method = cutout(rgba, backend)
    return mask, method
