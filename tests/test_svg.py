import math

import numpy as np
import pytest

from vector3d.core import affine
from vector3d.core.confidence import Confidence
from vector3d.svg.parser import parse_svg
from vector3d.svg.path_data import parse_path
from vector3d.svg.transform import parse_transform

SVG = '<svg xmlns="http://www.w3.org/2000/svg" {attrs}>{body}</svg>'


def doc(body, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return parse_svg(SVG.format(attrs=a, body=body), is_text=True)


def test_transform_order_is_left_to_right():
    m = parse_transform("translate(10 0) scale(2)")
    assert np.allclose(affine.apply(m, [[1, 1]]), [[12, 2]])


def test_rotate_about_center():
    m = parse_transform("rotate(90 1 1)")
    assert np.allclose(affine.apply(m, [[2, 1]]), [[1, 2]])


def test_relative_path_and_implicit_lineto():
    (c,) = parse_path("m10 10 20 0 0 20 z")
    assert c.closed
    assert np.allclose(c.anchor_points, [[10, 10], [30, 10], [30, 30], [10, 10]])


def test_smooth_cubic_reflects_control_point():
    (c,) = parse_path("M0 0 C0 10 10 10 10 0 S20 -10 20 0")
    assert np.allclose(c.segments[1].points[1], [10, -10])


def test_packed_arc_flags_and_arc_accuracy():
    (c,) = parse_path("M0 0a10 10 0 1110 10")  # flags "1" "1" packed
    pts = c.sample(64)
    centre = np.array([10, 0])  # large-arc sweep arc from (0,0) to (10,10), r=10
    assert np.allclose(np.linalg.norm(pts - centre, axis=1), 10, atol=3e-3)
    assert np.allclose(pts[-1], [10, 10])


def test_circle_is_exactly_round():
    d = doc('<circle cx="5" cy="5" r="4"/>', viewBox="0 0 10 10")
    pts = d.contours[0].sample(64)
    assert np.allclose(np.linalg.norm(pts - [5, 5], axis=1), 4, atol=2e-3)
    assert d.contours[0].primitive["type"] == "circle"


def test_nested_group_transforms_and_layers():
    d = doc('<g id="a" transform="translate(10 0)"><g id="b" transform="scale(2)">'
            '<line x1="0" y1="0" x2="1" y2="0"/></g></g>', viewBox="0 0 100 100")
    c = d.contours[0]
    assert np.allclose(c.anchor_points, [[10, 0], [12, 0]])
    assert c.layer_path == ("a", "b")


def test_pixels_are_not_physical():
    d = doc('<rect width="10" height="10"/>', width="100px", height="100px", viewBox="0 0 10 10")
    assert not d.scale.known


def test_document_units_with_drawing_scale_are_verified():
    d = doc('<rect width="10" height="10"/>', width="100mm", height="100mm", viewBox="0 0 10 10",
            data_v3d_drawing_scale="1:10")
    assert math.isclose(d.scale.meters_per_unit, 0.1)
    assert d.scale.confidence is Confidence.VERIFIED


def test_document_units_without_drawing_scale_are_estimated():
    d = doc('<rect width="10" height="10"/>', width="10cm", height="10cm", viewBox="0 0 10 10")
    assert math.isclose(d.scale.meters_per_unit, 0.01)
    assert d.scale.confidence is Confidence.ESTIMATED


def test_dimension_annotation_sets_scale_and_is_not_geometry():
    d = doc('<rect width="50" height="20"/>'
            '<line x1="0" y1="30" x2="50" y2="30" data-v3d-role="dimension" data-v3d-dimension="500mm"/>',
            viewBox="0 0 100 100")
    assert math.isclose(d.scale.meters_per_unit, 0.01)
    assert d.scale.source == "dimension"
    assert len(d.contours) == 1 and len(d.construction) == 1


def test_conflicting_scale_sources_warn():
    d = doc('<line x1="0" y1="0" x2="10" y2="0" data-v3d-dimension="1m"/>',
            viewBox="0 0 10 10", data_v3d_scale="0.5")
    assert d.scale.source == "explicit"
    assert any("disagrees" in w for w in d.warnings)


def test_hidden_elements_skipped_and_dashed_marked():
    d = doc('<line x1="0" y1="0" x2="1" y2="0" style="display:none"/>'
            '<line x1="0" y1="1" x2="1" y2="1" stroke-dasharray="2 1"/>', viewBox="0 0 10 10")
    assert len(d.contours) == 1
    assert d.contours[0].semantic["line_style"] == "dashed"


def test_use_resolves_symbol_with_offset():
    d = doc('<defs><symbol id="s"><line x1="0" y1="0" x2="1" y2="0"/></symbol></defs>'
            '<use href="#s" x="5" y="5"/>', viewBox="0 0 10 10")
    assert np.allclose(d.contours[0].anchor_points, [[5, 5], [6, 5]])


def test_semantic_part_inherited():
    d = doc('<g data-v3d-part="wheel"><circle cx="0" cy="0" r="1" id="rim"/></g>', viewBox="0 0 10 10")
    assert d.contours[0].semantic["data-v3d-part"] == "wheel"
    assert d.contours[0].element_id == "rim"


def test_bad_path_is_reported_not_fatal():
    d = doc('<path d="L 1 2"/><line x1="0" y1="0" x2="1" y2="0"/>', viewBox="0 0 10 10")
    assert len(d.contours) == 1
    assert d.warnings


@pytest.mark.parametrize("d", ["M0 0 L", "M 0 0 A 1 1 0 2 0 1 1"])
def test_malformed_path_raises(d):
    with pytest.raises(ValueError):
        parse_path(d)
