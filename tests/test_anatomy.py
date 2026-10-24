"""Anatomy step (RTMW + MHR) pieces that run without WSL or a GPU."""
import numpy as np
from PIL import Image

from vector3d.lab import anatomy

GOOD = {"labels": dict(anatomy.EXPECTED_LABELS),
        "hands": {"left": {"label": "ESTIMATED", "fingers_model": 5, "fingers_detected": 5, "fingertips_checked": 5,
                           "fingertips_matched": 5, "finger_pairs_closer_than_1cm": 0},
                  "right": {"label": "ESTIMATED", "fingers_model": 5, "fingers_detected": 3, "fingertips_checked": 3,
                            "fingertips_matched": 2, "finger_pairs_closer_than_1cm": 0}}}


def test_wsl_path():
    assert anatomy.wsl_path(r"C:\adbosp-vector3d-engine\bench\x.png") == "/mnt/c/adbosp-vector3d-engine/bench/x.png"


def test_labels_ok_and_fit_cannot_claim_verified():
    assert anatomy.check_labels(GOOD) == []
    bad = {"labels": {**anatomy.EXPECTED_LABELS, "fit": "VERIFIED", "mesh": "MEASURED"},
           "hands": {"left": {"label": "VERIFIED"}}}
    problems = anatomy.check_labels(bad)
    assert any(p.startswith("fit:") for p in problems)
    assert any(p.startswith("mesh:") for p in problems)
    assert any(p.startswith("hands.left") for p in problems)


def test_finger_summary_only_counts_checked_tips():
    s = anatomy.finger_summary(GOOD)
    assert s["left"]["correct"] is True
    assert s["right"] == {"fingers_detected": 3, "checked": 3, "matched": 2, "correct": False}
    none_checked = {"hands": {"left": {**GOOD["hands"]["left"], "fingertips_checked": 0, "fingertips_matched": 0}}}
    assert anatomy.finger_summary(none_checked)["left"]["correct"] is False


def test_blocked_without_environment(tmp_path):
    img = tmp_path / "p.png"
    Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(img)
    step = anatomy.AnatomyStep(distro="no-such-distro-v3d", python="/nonexistent/python", assets="/nonexistent")
    r = step.run(img, tmp_path / "out")
    assert r["status"] == "blocked" and r["reason"]
    assert not list((tmp_path / "out").rglob("*.glb"))
