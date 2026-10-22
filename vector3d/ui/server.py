"""Local web UI: a stdlib HTTP server exposing the engine as a JSON API.

  GET  /                 single-page interface (static/index.html)
  GET  /api/stages       pipeline stage registry
  GET  /api/capabilities whether the local AI 3D model (TripoSG) can run
  GET  /api/examples     bundled example SVGs, keyed by view
  POST /api/inspect      {"views": {"front": VIEW, ...}, "tolerance": 0.01}
  POST /api/generate     {"views": {...}, "options": {"mode": "auto|ai|drawing|photo", "resolution": 128,
                          "smooth_iterations": 10, "hints": {...}, "thickness": 0.6, "detail": 0.7,
                          "back": "symmetric|flat", "size_m": 1.0, "depth_backend": "auto"}}
                         -> {"job": id}; the job runs in a background thread
  GET  /api/jobs/<id>    step-by-step progress, then metadata
  GET  /api/jobs/<id>/model.glb | model.obj | metadata.json | mask.png | depth.png | normal.png

VIEW is SVG text, or {"name": "front.png", "data": "<base64>"} for any file (SVG, PNG, JPG).
"""
from __future__ import annotations

import base64
import binascii
import json
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree.ElementTree import ParseError

from ..generate import GenerateOptions, generate, load_view, plan_steps, resolve_mode
from ..normalize.normalizer import normalize_views
from ..normalize.views import ViewName
from ..pipeline import STAGES
from ..reconstruct.hints import validate_hints
from ..report import geometry, summary

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
MAX_BODY = 60 * 1024 * 1024
MAX_JOBS = 20


def decode_views(payload: dict) -> dict[str, tuple[str, bytes]]:
    views = payload.get("views") or {}
    if not isinstance(views, dict) or not views:
        raise ValueError("no views supplied")
    out = {}
    for name, item in views.items():
        view = ViewName(str(name).lower()).value
        if isinstance(item, str):
            out[view] = (f"{view}.svg", item.encode("utf-8"))
        else:
            try:
                out[view] = (str(item.get("name") or view), base64.b64decode(item["data"], validate=True))
            except (KeyError, TypeError, binascii.Error) as exc:
                raise ValueError(f"{view}: expected SVG text or {{name, data(base64)}}") from exc
    return out


def _load_all(files: dict[str, tuple[str, bytes]]):
    docs = {}
    for view, (name, data) in files.items():
        try:
            docs[view] = load_view(view, name, data)
        except ParseError as exc:
            raise ValueError(f"{view}: invalid SVG ({exc})") from exc
        except OSError as exc:  # PIL: not an image it can read
            raise ValueError(f"{view}: unreadable image ({exc})") from exc
    return docs


def inspect(payload: dict) -> dict:
    docs = _load_all(decode_views(payload))
    scene = normalize_views(docs, tolerance=float(payload.get("tolerance", 0.01)))
    return {"summary": summary(scene), "geometry": geometry(scene)}


class Job:
    def __init__(self, files: dict[str, tuple[str, bytes]], options: GenerateOptions):
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self.status = "queued"
        self.error: str | None = None
        self.mode = resolve_mode(files, options.mode)
        self.steps = [{"key": k, "title": t, "status": "pending", "detail": ""} for k, t in plan_steps(self.mode)]
        self.result = None
        self._files, self._options = files, options
        self._lock = threading.Lock()

    def _progress(self, key: str, status: str, detail: str) -> None:
        with self._lock:
            for s in self.steps:
                if s["key"] == key:
                    s["status"], s["detail"] = status, detail

    def run(self) -> None:
        self.status = "running"
        try:
            self.result = generate(self._files, self._options, self._progress)
            self.status = "done"
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self.status = "error"
        finally:
            self._files = {}

    def snapshot(self) -> dict:
        with self._lock:
            out = {"id": self.id, "status": self.status, "error": self.error, "mode": self.mode,
                   "steps": [dict(s) for s in self.steps]}
        if self.result is not None:
            out["metadata"] = self.result.metadata
            out["files"] = {"glb": f"/api/jobs/{self.id}/model.glb", "obj": f"/api/jobs/{self.id}/model.obj",
                            "metadata": f"/api/jobs/{self.id}/metadata.json"}
            out["images"] = {k.removesuffix(".png"): f"/api/jobs/{self.id}/{k}" for k in self.result.images}
        return out


JOBS: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def start_job(payload: dict) -> Job:
    files = decode_views(payload)
    _load_all(files)  # fail fast on unreadable inputs, before a job exists
    opts = payload.get("options") or {}
    clamp = lambda v, lo, hi: min(max(v, lo), hi)  # noqa: E731
    options = GenerateOptions(
        resolution=int(clamp(int(opts.get("resolution", 128)), 16, 384)),
        smooth_iterations=int(clamp(int(opts.get("smooth_iterations", 10)), 0, 50)),
        hints=opts.get("hints") or None,
        mode=str(opts.get("mode", "auto")),
        depth_backend=str(opts.get("depth_backend", "auto")),
        thickness=float(clamp(float(opts.get("thickness", 0.6)), 0.05, 3.0)),
        detail=float(clamp(float(opts.get("detail", 0.7)), 0.0, 1.0)),
        back=str(opts.get("back", "symmetric")),
        size_m=float(clamp(float(opts.get("size_m", 1.0)), 1e-3, 1e4)),
        ai_profile=str(opts.get("ai_profile", "balanced")),
    )
    if options.ai_profile not in ("fast", "balanced", "quality"):
        raise ValueError("ai_profile must be fast, balanced or quality")
    if options.back not in ("symmetric", "flat"):
        raise ValueError("back must be symmetric or flat")
    if options.hints:
        validate_hints(options.hints)
    job = Job(files, options)
    with _jobs_lock:
        if len(JOBS) >= MAX_JOBS:
            oldest = min(JOBS.values(), key=lambda j: j.created)
            JOBS.pop(oldest.id, None)
        JOBS[job.id] = job
    threading.Thread(target=job.run, daemon=True).start()
    return job


def examples() -> dict[str, str]:
    out = {}
    if EXAMPLES.is_dir():
        for v in ViewName:
            p = EXAMPLES / f"box_{v.value}.svg"
            if p.exists():
                out[v.value] = p.read_text(encoding="utf-8")
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "Vector3D/0.1"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/stages":
            self._json(200, [asdict(s) for s in STAGES])
        elif path == "/api/capabilities":
            from ..generate import ai_available

            ok, why = ai_available()
            self._json(200, {"ai": ok, "ai_model": "VAST-AI/TripoSG", "ai_reason": why})
        elif path == "/api/examples":
            self._json(200, examples())
        elif path.startswith("/api/jobs/"):
            self._job_get(path[len("/api/jobs/"):])
        else:
            self._json(404, {"error": "not found"})

    def _job_get(self, rest: str) -> None:
        job_id, _, name = rest.partition("/")
        job = JOBS.get(job_id)
        if job is None:
            self._json(404, {"error": "unknown job"})
        elif not name:
            self._json(200, job.snapshot())
        elif job.result is None:
            self._json(409, {"error": f"job is {job.status}"})
        elif name == "model.glb":
            self._send(200, job.result.glb, "model/gltf-binary")
        elif name == "model.obj":
            self._send(200, job.result.obj.encode("utf-8"), "text/plain; charset=utf-8")
        elif name == "metadata.json":
            self._json(200, job.result.metadata)
        elif name in job.result.images:
            self._send(200, job.result.images[name], "image/png")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/api/inspect", "/api/generate"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length else 400, {"error": "bad request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if path == "/api/inspect":
                self._json(200, inspect(payload))
            else:
                self._json(202, {"job": start_job(payload).id})
        except (ValueError, KeyError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # report, don't kill the server
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = make_server(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Vector3D UI: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
