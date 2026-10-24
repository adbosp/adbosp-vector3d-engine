"""Anatomy step (RTMW + MHR) pieces that run without WSL or a GPU."""
import shlex

import numpy as np
import pytest
from PIL import Image

from vector3d.lab import anatomy


def _hand(checked, matched, detected=None):
    return {"label": "ESTIMATED", "fingers_model": 5, "fingers_detected": checked if detected is None else detected,
            "fingertips_checked": checked, "fingertips_matched": matched, "finger_axis_pairs_closer_than_1cm": 0}


GOOD = {"labels": dict(anatomy.EXPECTED_LABELS), "hands": {"left": _hand(5, 5), "right": _hand(3, 2)}}


def test_wsl_path():
    assert anatomy.wsl_path(r"C:\adbosp-vector3d-engine\bench\x.png") == "/mnt/c/adbosp-vector3d-engine/bench/x.png"


def test_script_quotes_paths_with_spaces(tmp_path):
    d = tmp_path / "my runs" / "it's here"
    img = d / "person 1.png"
    step = anatomy.AnatomyStep(python="~/env/bin/python", assets="~/mhr/assets")
    s = step.script(img, d, d / "result.json", 600, 0)
    args = shlex.split(s)
    assert args[args.index("--image") + 1] == anatomy.wsl_path(img)
    assert args[args.index("--out-dir") + 1] == anatomy.wsl_path(d)
    assert args[0] == "~/env/bin/python"            # configured paths stay unquoted so ~ expands in bash


def test_labels_ok_and_fit_cannot_claim_verified():
    assert anatomy.check_labels(GOOD) == []
    bad = {"labels": {**anatomy.EXPECTED_LABELS, "fit": "VERIFIED", "mesh": "MEASURED"},
           "hands": {"left": {"label": "VERIFIED"}}}
    problems = anatomy.check_labels(bad)
    assert any(p.startswith("fit:") for p in problems)
    assert any(p.startswith("mesh:") for p in problems)
    assert any(p.startswith("hands.left") for p in problems)


@pytest.mark.parametrize("checked,matched,cov,partial,full", [
    (0, 0, 0.0, False, False),
    (1, 1, 0.2, True, False),          # one matching tip is not a consistent hand
    (4, 4, 0.8, True, False),
    (5, 5, 1.0, True, True),
    (5, 4, 1.0, False, False),
])
def test_finger_summary_coverage(checked, matched, cov, partial, full):
    s = anatomy.finger_summary({"hands": {"left": _hand(checked, matched)}})["left"]
    assert s["coverage"] == cov
    assert s["detected_tips_consistent"] is partial and s["all_five_tips_consistent"] is full
    assert "correct" not in s and s["anatomical_correctness"].startswith("not measured")


def test_finger_summary_reads_old_key():
    old = {"hands": {"left": {**_hand(5, 5), "finger_pairs_closer_than_1cm": 2}}}
    del old["hands"]["left"]["finger_axis_pairs_closer_than_1cm"]
    assert anatomy.finger_summary(old)["left"]["finger_axis_pairs_closer_than_1cm"] == 2


def test_blocked_without_environment(tmp_path):
    img = tmp_path / "p.png"
    Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(img)
    step = anatomy.AnatomyStep(distro="no-such-distro-v3d", python="/nonexistent/python", assets="/nonexistent")
    r = step.run(img, tmp_path / "out")
    assert r["status"] == "blocked" and r["reason"]
    assert not list((tmp_path / "out").rglob("*.glb"))
