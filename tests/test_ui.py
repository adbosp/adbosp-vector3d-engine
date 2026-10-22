import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from vector3d.ui.server import examples, inspect, make_server

EX = Path(__file__).resolve().parent.parent / "examples"


def test_inspect_places_views_on_glass_box():
    out = inspect({"views": examples()})
    assert out["summary"]["units"] == "m"
    geo = out["geometry"]
    # front view sits on the +Z face (box depth 3 m -> z = 1.5)
    front = np.array(geo["front"]["contours"][0]["world"])
    assert np.allclose(front[:, 2], 1.5)
    # right view (object's right side, camera at -X) sits on the -X face
    right = np.array(geo["right"]["contours"][0]["world"])
    assert np.allclose(right[:, 0], -1.0)
    assert any(c["dashed"] for c in geo["right"]["contours"])
    assert any(c["kind"] == "construction" for c in geo["top"]["contours"])


@pytest.mark.parametrize("payload", [{}, {"views": {"side": "<svg/>"}}, {"views": {"front": "<svg"}}])
def test_inspect_rejects_bad_input(payload):
    with pytest.raises(ValueError):
        inspect(payload)


@pytest.fixture
def server():
    httpd = make_server(port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def test_http_roundtrip(server):
    html = urllib.request.urlopen(server + "/").read().decode()
    assert "Vector3D Engine" in html
    stages = json.load(urllib.request.urlopen(server + "/api/stages"))
    assert stages[0]["key"] == "parse"
    out = _post(server + "/api/inspect", {"views": examples()})
    assert set(out["geometry"]) == {"front", "right", "top"}
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(server + "/api/inspect", {"views": {}})
    assert err.value.code == 400
