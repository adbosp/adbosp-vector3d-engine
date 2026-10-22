"""Local web UI: a stdlib HTTP server exposing the engine as a JSON API.

  GET  /                 single-page interface (static/index.html)
  GET  /api/stages       pipeline stage registry
  GET  /api/examples     bundled example SVGs, keyed by view
  POST /api/inspect      {"views": {"front": "<svg ...>", ...}, "tolerance": 0.01}
"""
from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree.ElementTree import ParseError

from ..normalize.normalizer import normalize_views
from ..normalize.views import ViewName
from ..pipeline import STAGES
from ..report import geometry, summary
from ..svg.parser import parse_svg

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
MAX_BODY = 20 * 1024 * 1024


def inspect(payload: dict) -> dict:
    views = payload.get("views") or {}
    if not isinstance(views, dict) or not views:
        raise ValueError("no views supplied")
    docs = {}
    for name, text in views.items():
        view = ViewName(str(name).lower())
        try:
            docs[view] = parse_svg(text, is_text=True)
        except ParseError as exc:
            raise ValueError(f"{view.value}: invalid SVG ({exc})") from exc
        docs[view].source = f"{view.value}.svg"
    scene = normalize_views(docs, tolerance=float(payload.get("tolerance", 0.01)))
    return {"summary": summary(scene), "geometry": geometry(scene)}


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
        elif path == "/api/examples":
            self._json(200, examples())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/inspect":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length else 400, {"error": "bad request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            self._json(200, inspect(payload))
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
