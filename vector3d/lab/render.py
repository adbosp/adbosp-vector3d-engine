"""Orthographic software rasterizer for previews, silhouettes and metrics.

Cameras use the engine's world frame (+Y up, object faces +Z). A camera is
an orthonormal basis plus a framing box, so the same camera renders the
reference silhouette and the candidate mesh pixel for pixel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

from ..normalize.views import CAMERAS, ViewName
from ..reconstruct.surface import Mesh


@dataclass(frozen=True)
class OrthoCamera:
    name: str
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    toward: tuple[float, float, float]   # unit vector from the object toward the camera
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_size: float = 1.0               # world half-width of the square image
    size: int = 256                      # image side in pixels

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OrthoCamera":
        return OrthoCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) world -> (N,3) pixel x, pixel y (down), depth toward the camera."""
        p = np.asarray(xyz, dtype=float) - np.array(self.center)
        b = np.array([self.right, self.up, self.toward], dtype=float)
        u, v, d = (p @ b.T).T
        s = self.size / (2 * self.half_size)
        return np.column_stack([(u + self.half_size) * s, (self.half_size - v) * s, d])


def view_camera(view: str, center=(0.0, 0.0, 0.0), half_size=1.0, size=256) -> OrthoCamera:
    cam = CAMERAS[ViewName(view)]
    return OrthoCamera(view, tuple(map(float, cam.right)), tuple(map(float, cam.up)),
                       tuple(map(float, cam.position_dir)), tuple(center), half_size, size)


def orbit_camera(azimuth_deg: float, elevation_deg: float = 0.0, center=(0.0, 0.0, 0.0),
                 half_size=1.0, size=256, name: str | None = None) -> OrthoCamera:
    """Azimuth 0 = front (+Z), 90 = the object's left side (+X)."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    toward = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.array([np.cos(a), 0.0, -np.sin(a)])
    up = np.cross(toward, right)
    return OrthoCamera(name or f"az{azimuth_deg:g}_el{elevation_deg:g}", tuple(right), tuple(up),
                       tuple(toward), tuple(center), half_size, size)


def frame(mesh_or_points, margin: float = 1.1) -> tuple[tuple[float, float, float], float]:
    """Center and half_size that fit the object in every orthographic view."""
    v = mesh_or_points.vertices if isinstance(mesh_or_points, Mesh) else np.asarray(mesh_or_points)
    lo, hi = v.min(axis=0), v.max(axis=0)
    c = (lo + hi) / 2
    r = float(np.linalg.norm(v - c, axis=1).max())
    return tuple(c.tolist()), r * margin


@njit(cache=True)
def _raster(px, faces, size, depth, face_id):
    for fi in range(faces.shape[0]):
        a, b, c = faces[fi, 0], faces[fi, 1], faces[fi, 2]
        x0, y0, z0 = px[a, 0], px[a, 1], px[a, 2]
        x1, y1, z1 = px[b, 0], px[b, 1], px[b, 2]
        x2, y2, z2 = px[c, 0], px[c, 1], px[c, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                sx, sy = x + 0.5, y + 0.5
                w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) / area
                w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > depth[y, x]:
                    depth[y, x] = z
                    face_id[y, x] = fi


@dataclass
class Render:
    camera: OrthoCamera
    depth: np.ndarray    # (S,S) depth toward camera, -inf where empty
    face_id: np.ndarray  # (S,S) int, -1 where empty

    @property
    def mask(self) -> np.ndarray:
        return self.face_id >= 0


def render(mesh: Mesh, camera: OrthoCamera) -> Render:
    s = camera.size
    px = camera.project(mesh.vertices)
    depth = np.full((s, s), -np.inf)
    fid = np.full((s, s), -1, dtype=np.int64)
    _raster(px, mesh.faces.astype(np.int64), s, depth, fid)
    return Render(camera, depth, fid)


def silhouette(mesh: Mesh, camera: OrthoCamera) -> np.ndarray:
    return render(mesh, camera).mask


def shade(mesh: Mesh, r: Render, base=(0.78, 0.80, 0.84), key=None) -> np.ndarray:
    """RGBA uint8 preview: headlight Lambert plus a key light, two-sided.
    key: key-light direction as weights on the camera's (right, up, toward); default (0.4, 0.5, 0.6)."""
    v = mesh.vertices[mesh.faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    cam = r.camera
    toward = np.array(cam.toward)
    kr, ku, kt = (0.4, 0.5, 0.6) if key is None else key
    key = kt * toward + ku * np.array(cam.up) + kr * np.array(cam.right)
    key /= np.linalg.norm(key)
    n = fn[np.maximum(r.face_id, 0)]
    head = np.abs(n @ toward)
    k = np.clip(n @ key, 0, 1)
    if mesh.colors is not None:
        cols = mesh.colors[mesh.faces].mean(axis=1)[np.maximum(r.face_id, 0)]
    else:
        cols = np.broadcast_to(np.array(base), n.shape)
    lum = (0.25 + 0.45 * head + 0.35 * k)[..., None]
    rgb = np.clip(cols * lum, 0, 1)
    out = np.zeros((*r.mask.shape, 4), np.uint8)
    out[..., :3] = (rgb * 255).astype(np.uint8)
    out[..., 3] = np.where(r.mask, 255, 0)
    out[~r.mask, :3] = 0
    return out


def contact_sheet(images: list[np.ndarray], labels: list[str] | None = None, bg=(245, 245, 247)) -> np.ndarray:
    """Tile RGBA renders side by side on a flat background (RGB uint8)."""
    from PIL import Image, ImageDraw

    s = images[0].shape[0]
    sheet = Image.new("RGB", (s * len(images), s + 18), bg)
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(images):
        tile = Image.fromarray(im, "RGBA")
        sheet.paste(tile, (i * s, 18), tile)
        if labels:
            draw.text((i * s + 4, 3), labels[i], fill=(40, 40, 40))
    return np.asarray(sheet)


def turntable(mesh: Mesh, n: int = 8, elevation: float = 15.0, size: int = 256) -> list[tuple[str, np.ndarray]]:
    center, half = frame(mesh)
    out = []
    for i in range(n):
        cam = orbit_camera(360.0 * i / n, elevation, center, half, size)
        out.append((cam.name, shade(mesh, render(mesh, cam))))
    return out
