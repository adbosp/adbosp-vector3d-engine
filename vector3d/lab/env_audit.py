"""Hardware and software audit, run before installing any model.

Reports what is measured (GPU, VRAM, driver, CUDA, RAM, disk, compilers)
and which candidate models the official requirements allow. A model is
"fits_on_paper" from the published VRAM figure only; it becomes "measured"
after a real run.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Official minimums from research/model_matrix.md (GB, None = not published).
MODEL_REQUIREMENTS = {
    "triposg": {"vram_gb": 8, "os": "any (not stated)", "license": "MIT"},
    "hunyuan3d-2mini/2mv (shape)": {"vram_gb": 6, "os": "win/mac/linux", "license": "Tencent Hunyuan community (EU/UK/KR excluded)"},
    "hunyuan3d-2.1 (shape)": {"vram_gb": 10, "os": "win/mac/linux", "license": "Tencent Hunyuan community (EU/UK/KR excluded)"},
    "hunyuan3d-2.1 (shape+PBR)": {"vram_gb": 29, "os": "win/mac/linux", "license": "Tencent Hunyuan community (EU/UK/KR excluded)"},
    "trellis.2": {"vram_gb": 24, "os": "linux", "license": "MIT"},
    "pixal3d": {"vram_gb": None, "os": "not stated (TRELLIS.2 stack)", "license": "MIT"},
}


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpus() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    header = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    rows = []
    for line in out.strip().splitlines():
        name, total, free, drv = [x.strip() for x in line.split(",")]
        rows.append({"name": name, "vram_total_mb": float(total), "vram_free_mb": float(free),
                     "driver": drv, "driver_cuda": m.group(1) if m else None})
    return rows


def ram_gb() -> float | None:
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS()
        s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 2**30
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def torch_info(python: str) -> dict:
    code = ("import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else None}))")
    out = _run([python, "-c", code], timeout=120)
    return json.loads(out.strip().splitlines()[-1]) if out else {"torch": None, "note": f"torch not importable in {python}"}


def audit(python: str | None = None, disk_path: str = ".") -> dict:
    g = gpus()
    vram = max((x["vram_total_mb"] for x in g), default=0) / 1024
    is_wsl = "microsoft" in platform.release().lower()
    rep = {
        "os": platform.platform(), "wsl": is_wsl, "python": sys.version.split()[0], "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "ram_gb": ram_gb(),
        "disk_free_gb": shutil.disk_usage(disk_path).free / 2**30,
        "gpus": g, "torch": torch_info(python or sys.executable),
        "tools": {t: bool(shutil.which(t)) for t in ("git", "nvcc", "cl", "gcc", "blender", "wsl")},
        "models": {},
    }
    linux = sys.platform.startswith("linux")
    for name, req in MODEL_REQUIREMENTS.items():
        need = req["vram_gb"]
        if not g:
            verdict = "no NVIDIA GPU detected"
        elif need is None:
            verdict = "unknown: no published VRAM figure, must be measured"
        elif vram >= need:
            verdict = "fits_on_paper"
        else:
            verdict = f"does_not_fit ({need} GB needed, {vram:.1f} GB present)"
        if req["os"] == "linux" and not linux:
            verdict += "; Linux-only, use WSL2 or cloud"
        rep["models"][name] = {**req, "verdict": verdict}
    notes = []
    if g and not rep["tools"]["nvcc"]:
        notes.append("nvcc missing: TripoSG's diso and Hunyuan texture extensions need the CUDA toolkit to compile")
    if sys.platform == "win32" and not rep["tools"]["cl"]:
        notes.append("MSVC cl.exe not on PATH: CUDA extensions on Windows need the VS Build Tools developer prompt")
    if rep["disk_free_gb"] < 30:
        notes.append("less than 30 GB free: weights + envs for one model need roughly 15-25 GB")
    rep["notes"] = notes
    return rep
