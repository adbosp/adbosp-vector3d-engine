#!/usr/bin/env bash
# TRELLIS v1 (microsoft/TRELLIS-image-large, MIT) geometry-only environment on WSL2, separate from every other env.
# Only what the image -> mesh path needs: torch 2.4.0+cu121, xformers, spconv, kaolin (FlexiCubes helper), utils3d.
# nvdiffrast (NVIDIA Source Code License, non-commercial) is NOT installed: it is only used for textured GLB
# post-processing, which this engine does not use. See research/trellis/LICENCES.md. No sudo.
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
cd $HOME/ai3d
[ -x bin/micromamba ] || { echo "run scripts/setup_hi3d_env.sh or setup_anatomy_env.sh first (micromamba)"; exit 1; }
[ -d TRELLIS ] || git clone -q --recurse-submodules https://github.com/microsoft/TRELLIS
git -C TRELLIS checkout -q 442aa1e
[ -d $MAMBA_ROOT_PREFIX/envs/trellis ] || bin/micromamba create -y -n trellis -c conda-forge python=3.10 gcc_linux-64=12 gxx_linux-64=12 git
PY=$MAMBA_ROOT_PREFIX/envs/trellis/bin/python
$PY -m pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install -q xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
# numpy 1.x for the torch 2.4 / spconv / kaolin stack; opencv and plyfile pinned to releases that accept it
$PY -m pip install -q numpy==1.26.4 opencv-python-headless==4.10.0.84 "plyfile<1.1" pillow imageio tqdm easydict \
    scipy ninja trimesh transformers huggingface_hub spconv-cu120 rembg==2.0.69 onnxruntime==1.20.1   # rembg: imported by TRELLIS at load, never called (inputs carry alpha)
# open3d (MIT) is evaluated at import by TRELLIS's text pipeline; it needs libgomp from conda-forge (LD_LIBRARY_PATH is set by backends/trellis.py)
bin/micromamba install -y -q -n trellis -c conda-forge libgomp
$PY -m pip install -q open3d==0.18.0 numpy==1.26.4
$PY -m pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
$PY -m pip install -q kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
mkdir -p weights
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('microsoft/TRELLIS-image-large', local_dir='weights/TRELLIS-image-large'))"
$PY - <<'EOF'
import torch, xformers, spconv, kaolin
print("torch", torch.__version__, torch.cuda.is_available(), "xformers", xformers.__version__, "kaolin", kaolin.__version__)
EOF
du -sh weights/TRELLIS-image-large
$PY -m pip freeze > $HOME/ai3d/trellis_pip_freeze.txt
echo SETUP_DONE
