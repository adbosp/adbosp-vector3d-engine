from pathlib import Path

import numpy as np
import pytest

from vector3d import CAMERAS, OPPOSITE, Confidence, ViewName, normalize_views, parse_svg
from vector3d.normalize.views import mirror_to_opposite, shared_axes

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("view", list(ViewName))
def test_camera_basis_is_right_handed_and_looks_at_object(view):
    cam = CAMERAS[view]
    b = cam.basis
    assert np.allclose(b.T @ b, np.eye(3))
    assert np.isclose(np.linalg.det(b), 1.0)
    # right x up = toward camera, i.e. the camera looks along -position_dir
    assert np.allclose(np.cross(cam.right, cam.up), cam.position_dir)


def test_opposite_pairs_share_axes_and_mirror_horizontal():
    for a, b in OPPOSITE.items():
        assert len(shared_axes(a, b)) == 2
        assert np.allclose(CAMERAS[a].forward, -CAMERAS[b].forward)
    uv = np.array([[0.3, 0.2]])
    assert np.allclose(mirror_to_opposite(ViewName.FRONT, uv), [[-0.3, 0.2]])
    assert np.allclose(mirror_to_opposite(ViewName.TOP, uv), [[0.3, -0.2]])


def test_world_axes_per_view():
    assert CAMERAS[ViewName.FRONT].axis_of(0) == (0, 1)
    # object's right side is at -X; seen from there, +u points toward the front (+Z)
    assert CAMERAS[ViewName.RIGHT].depth_axis == 0
    assert CAMERAS[ViewName.RIGHT].position_dir == (-1, 0, 0)
    assert CAMERAS[ViewName.RIGHT].axis_of(0) == (2, 1)
    assert CAMERAS[ViewName.LEFT].axis_of(0) == (2, -1)
    assert CAMERAS[ViewName.TOP].axis_of(1) == (2, -1)


def test_box_example_normalizes_to_meters():
    docs = {v: parse_svg(EX / f"box_{v}.svg") for v in ("front", "right", "top")}
    scene = normalize_views(docs)
    assert scene.units == "m"
    assert all(c.ok for c in scene.checks) and len(scene.checks) == 3
    ext = scene.views[ViewName.RIGHT].world_extent()
    assert np.allclose(ext[2], (-1.5, 1.5)) and np.allclose(ext[1], (-0.5, 0.5))
    assert scene.views[ViewName.RIGHT].scale.confidence is Confidence.ESTIMATED
    assert scene.views[ViewName.TOP].scale.source == "dimension"


def test_svg_y_down_becomes_v_up():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" data-v3d-scale="1">'
           '<line x1="0" y1="0" x2="0" y2="10"/><circle id="v3d-origin" cx="0" cy="10" r="0.1"/></svg>')
    scene = normalize_views({"front": parse_svg(svg, is_text=True)})
    nv = scene.views[ViewName.FRONT]
    assert nv.origin_source == "marker"
    assert np.allclose(nv.contours[0].anchor_points, [[0, 10], [0, 0]])
    world = nv.camera.to_world(nv.contours[0].anchor_points)
    assert np.allclose(world, [[0, 10, 0], [0, 0, 0]])


def test_unscaled_views_stay_in_drawing_units():
    s = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="{w}" height="{h}"/></svg>'
    docs = {"front": parse_svg(s.format(w=40, h=20), is_text=True),
            "left": parse_svg(s.format(w=80, h=40), is_text=True)}
    scene = normalize_views(docs)
    assert scene.units == "drawing_units"
    # left drawn at 2x: normalized to front's units, so Z extent is 40, Y extent 20
    ext = scene.views[ViewName.LEFT].world_extent()
    assert np.isclose(ext[2][1] - ext[2][0], 40) and np.isclose(ext[1][1] - ext[1][0], 20)


def test_mismatched_extents_are_flagged():
    s = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" data-v3d-scale="0.01"><rect width="{w}" height="{h}"/></svg>'
    docs = {"front": parse_svg(s.format(w=40, h=20), is_text=True),
            "back": parse_svg(s.format(w=40, h=25), is_text=True)}
    scene = normalize_views(docs)
    bad = [c for c in scene.checks if not c.ok]
    assert len(bad) == 1 and bad[0].world_axis == "Y"


def test_declared_view_must_match():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" data-v3d-view="top"><rect width="1" height="1"/></svg>'
    with pytest.raises(ValueError):
        normalize_views({"front": parse_svg(svg, is_text=True)})
