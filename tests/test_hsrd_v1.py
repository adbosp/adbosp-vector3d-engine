"""hsrd_v1 pieces that need no dataset: frame conversion, person split, texture sampling, light weights."""
import numpy as np
from PIL import Image

from vector3d.lab import hsrd_v1


def test_engine_frame_is_y_up_front_plus_z_and_a_rotation():
    up, front, right_of_person = hsrd_v1.to_engine_frame(np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]))
    assert np.allclose(up, [0, 1, 0]) and np.allclose(front, [0, 0, 1])
    r = hsrd_v1.to_engine_frame(np.eye(3))                  # rows are the images of the basis vectors
    assert np.isclose(np.linalg.det(r), 1.0)                # proper rotation, no mirror
    assert np.allclose(right_of_person, [1, 0, 0])          # (-1,0,0) maps to +X: the person's right side stays at -X when facing +Z


def test_split_is_by_person_and_fixed():
    persons = [f"HSR{i:04d}" for i in range(10)]
    a, b = hsrd_v1.split(persons), hsrd_v1.split(list(reversed(persons)))
    assert a == b and len(a["test"]) == 2 and len(a["train"]) == 8
    assert not set(a["test"]) & set(a["train"])


def test_ground_puts_feet_on_zero_and_centres():
    v = hsrd_v1.ground(np.array([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]]))
    assert np.allclose(v.min(0)[1], 0) and np.allclose((v.min(0) + v.max(0))[[0, 2]] / 2, 0)


def test_texture_colours_samples_uv_with_v_flipped():
    tex = np.zeros((4, 4, 3), np.uint8)
    tex[0, 0] = (255, 0, 0)        # top-left texel = uv (0, 1)
    tex[3, 3] = (0, 0, 255)        # bottom-right texel = uv (1, 0)
    c = hsrd_v1.texture_colours(np.array([[0.0, 1.0], [1.0, 0.0]]), Image.fromarray(tex))
    assert np.allclose(c[0], [1, 0, 0]) and np.allclose(c[1], [0, 0, 1])
