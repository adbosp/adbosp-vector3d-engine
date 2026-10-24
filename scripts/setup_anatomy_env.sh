#!/usr/bin/env bash
# Anatomy step environment (WSL2 / Linux), separate from TripoSG and Hi3D:
#   RTMW whole-body keypoints (open-mmlab RTMPose project, Apache-2.0) via rtmlib on onnxruntime-gpu (no mmcv)
#   MHR body model (Meta, Apache-2.0 code; release assets carry their own LICENSE) for fitting, on CPU torch.
# conda-forge owns python, pytorch and pymomentum (no pip wheel of pymomentum exists for linux here);
# pip adds rtmlib and onnxruntime-gpu with its own CUDA 12 / cuDNN 9 wheels. No sudo.
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
mkdir -p $HOME/ai3d && cd $HOME/ai3d
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2
  python3 -c "import tarfile; tarfile.open('mm.tar.bz2').extract('bin/micromamba', '.')"
  rm mm.tar.bz2
fi
[ -d $MAMBA_ROOT_PREFIX/envs/anatomy ] || bin/micromamba create -y -n anatomy -c conda-forge \
  python=3.12 "pymomentum>=0.1.90" pytorch "trimesh>=4.8.3,<5" numpy pillow scipy
PY=$MAMBA_ROOT_PREFIX/envs/anatomy/bin/python
# rtmlib 0.0.16 also pulls CPU onnxruntime (1.31); the -gpu build installed after it is the one imported
# (CUDAExecutionProvider listed and used on ROPY-PC). Models it downloads on first use (sha256 on ROPY-PC):
#   rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx  bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054
#   yolox_m_8xb8-300e_humanart-c2c7a14a.onnx                 3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23
$PY -m pip install -q mhr==1.0.1 rtmlib==0.0.16
$PY -m pip install -q "onnxruntime-gpu[cuda,cudnn]==1.21.1" opencv-python-headless
mkdir -p $HOME/ai3d/mhr
if [ ! -f $HOME/ai3d/mhr/assets/compact_v6_1.model ]; then
  curl -Ls https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip -o $HOME/ai3d/mhr/assets.zip
  $PY -c "import zipfile; zipfile.ZipFile('$HOME/ai3d/mhr/assets.zip').extractall('$HOME/ai3d/mhr')"
fi
sha256sum $HOME/ai3d/mhr/assets.zip   # ROPY-PC: e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94
ls $HOME/ai3d/mhr/assets
$PY - <<'EOF'
import onnxruntime as ort
ort.preload_dlls()
import torch, rtmlib, pymomentum, mhr
print("torch", torch.__version__, "ort", ort.__version__, ort.get_available_providers())
print("mhr", mhr.__file__, "rtmlib", rtmlib.__file__)
EOF
$PY -m pip freeze > $HOME/ai3d/anatomy_pip_freeze.txt
echo SETUP_DONE
