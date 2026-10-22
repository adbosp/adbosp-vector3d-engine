import pytest

from vector3d.lab.backends import triposg


@pytest.fixture(autouse=True)
def no_machine_triposg(monkeypatch, tmp_path_factory):
    """Tests never see a TripoSG install on this machine (C:\\ai3d or the V3D_TRIPOSG_* variables
    start.cmd sets): with it, auto mode picks AI and real inference runs. Tests that need TripoSG set
    the variables to a stub themselves."""
    missing = tmp_path_factory.getbasetemp() / "no_triposg"
    monkeypatch.setattr(triposg, "DEFAULT_REPO", missing / "TripoSG")
    monkeypatch.setattr(triposg, "DEFAULT_PYTHON", missing / "python.exe")
    for var in ("V3D_TRIPOSG_DIR", "V3D_TRIPOSG_PYTHON", "V3D_TRIPOSG_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)
