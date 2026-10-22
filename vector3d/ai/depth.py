"""Monocular depth estimation for photos.

Default backend: Depth Anything V2 Small (Apache-2.0) as ONNX, run with
onnxruntime on CPU (or GPU providers when installed). The model is
downloaded once into ~/.cache/vector3d. Output is *relative inverse depth*
(larger = closer to the camera) with unknown scale and offset; callers must
treat it as AI_GENERATED shape information, never as measurements.

Fallback backend: silhouette inflation (no AI): a rounded cross-section
derived from the distance to the silhouette edge.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

DEFAULT_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
CACHE = Path(os.environ.get("V3D_CACHE", Path.home() / ".cache" / "vector3d"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DepthResult:
    depth: np.ndarray       # (H, W) float32 in [0, 1] inside the mask, 1 = nearest; 0 outside
    backend: str
    note: str = ""


def inflation(mask: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Rounded height in [0, 1] from the silhouette alone.

    Solves the Poisson problem  laplace(u) = -1  inside the mask with u = 0
    on the outline and returns sqrt(u), normalised. For a disc this is
    exactly a hemisphere; unlike a distance transform it has no creases
    along the medial axis.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    h, w = mask.shape
    k = min(1.0, max_side / max(h, w))
    small = mask if k == 1.0 else np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR)) > 127
    small = np.pad(small, 1)                      # guarantees a zero boundary ring
    idx = -np.ones(small.shape, dtype=np.int64)
    inside = np.argwhere(small)
    if len(inside) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    idx[small] = np.arange(len(inside))
    rows, cols, vals = [np.arange(len(inside))], [np.arange(len(inside))], [np.full(len(inside), 4.0)]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = idx[inside[:, 0] + dy, inside[:, 1] + dx]
        ok = nb >= 0
        rows.append(np.flatnonzero(ok)); cols.append(nb[ok]); vals.append(np.full(ok.sum(), -1.0))
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(inside), len(inside))).tocsr()
    u = np.zeros(small.shape, dtype=np.float64)
    u[small] = spsolve(A, np.ones(len(inside)))
    height = np.sqrt(np.clip(u[1:-1, 1:-1], 0, None))
    height /= height.max() or 1.0
    if k != 1.0:
        height = np.asarray(Image.fromarray(height.astype(np.float32)).resize((w, h), Image.BILINEAR))
    height = np.where(mask, height, 0).astype(np.float32)
    return height / (height.max() or 1.0)


def _normalize(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = raw[mask]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
    out = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    out[~mask] = 0
    return out


class DepthAnythingONNX:
    name = "depth-anything-v2-small"

    def __init__(self, model_path: str | Path | None = None, url: str | None = None, input_size: int = 518):
        self.model_path = Path(model_path or os.environ.get("V3D_DEPTH_MODEL") or CACHE / "depth_anything_v2_small.onnx")
        self.url = url or os.environ.get("V3D_DEPTH_URL") or DEFAULT_URL
        self.input_size = input_size
        self._session = None

    def ensure_model(self) -> Path:
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.model_path.with_suffix(".part")
            urllib.request.urlretrieve(self.url, tmp)
            if tmp.stat().st_size < 1_000_000:
                tmp.unlink()
                raise RuntimeError(f"download from {self.url} is too small to be a model")
            tmp.replace(self.model_path)
        return self.model_path

    def session(self):
        if self._session is None:
            import onnxruntime as ort

            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            self._session = ort.InferenceSession(str(self.ensure_model()), providers=providers)
        return self._session

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Raw relative inverse depth at the input resolution."""
        h, w = rgb.shape[:2]
        scale = self.input_size / max(h, w)
        nh = max(14, int(round(h * scale / 14)) * 14)
        nw = max(14, int(round(w * scale / 14)) * 14)
        x = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        sess = self.session()
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        out = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
        return np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR), dtype=np.float32)


def estimate_depth(rgb: np.ndarray, mask: np.ndarray, backend: str = "auto") -> DepthResult:
    """backend: auto | depth-anything | inflation."""
    if backend in ("auto", "depth-anything"):
        try:
            model = DepthAnythingONNX()
            return DepthResult(_normalize(model.predict(rgb), mask), model.name)
        except Exception as exc:  # missing onnxruntime, offline, bad model
            if backend == "depth-anything":
                raise
            return DepthResult(inflation(mask), "inflation",
                               f"AI depth unavailable ({type(exc).__name__}: {exc}); used silhouette inflation")
    if backend == "inflation":
        return DepthResult(inflation(mask), "inflation", "silhouette inflation (no AI)")
    raise ValueError(f"unknown depth backend {backend!r}")


def normal_map(height: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tangent-space style normal map (H, W, 3) uint8 from a height field: R=+x, G=+y (up), B=toward viewer."""
    gy, gx = np.gradient(height.astype(np.float32) * strength)
    n = np.dstack([-gx, gy, np.ones_like(gx)])            # image rows grow downward, so +y is -row
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    rgb = ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)
    rgb[~mask] = (128, 128, 255)
    return rgb
