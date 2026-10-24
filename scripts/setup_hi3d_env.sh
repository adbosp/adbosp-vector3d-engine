#!/usr/bin/env bash
# Hi3D v1.0 environment as built on ROPY-PC (WSL2 Ubuntu 24.04, RTX 3060) on 2026-10-24.
# Separate from TripoSG and the engine. No sudo: micromamba gives Python 3.10 and gcc,
# which Triton needs at run time (export CC, see the run line at the end).
# Lock: research/omniforge/env/hi3d_pip_freeze.txt. Revisions used:
#   code    github.com/hi3d-official/Hi3Dv1.0 @ 13657ad09afaa92e350b461a1cd66399e44a59c4
#   weights hi3d-official/Hi3Dv1.0 @ d502e938cbedc6e87aba8884614c5ca6d177a4b6
#           (sha256 transformer_512 8ba88311dccf2346..., transformer_1024 a2ca5e44dde51cac..., vae 605d207c77f8db0b...)
#   encoders facebook/dinov2-large @ 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c
#            openai/clip-vit-large-patch14 @ 32bd64288804d66eefd0ccbe215aa642df71cc41
# Licence: Hi3D Open License Agreement 1.0 (commercial use needs "Powered by Hi3D 1.0"; >200k MAU or
# >1M USD/month needs a separate licence). Read LICENSE in the weights folder before any product use.
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
mkdir -p $HOME/ai3d && cd $HOME/ai3d
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2      # no bzip2 on the box
  python3 -c "import tarfile; tarfile.open('mm.tar.bz2').extract('bin/micromamba', '.')"
  rm mm.tar.bz2
fi
[ -d Hi3Dv1.0 ] || git clone -q https://github.com/hi3d-official/Hi3Dv1.0
git -C Hi3Dv1.0 checkout -q 13657ad09afaa92e350b461a1cd66399e44a59c4
[ -d $MAMBA_ROOT_PREFIX/envs/hi3d ] || bin/micromamba create -y -n hi3d -c conda-forge python=3.10 gcc_linux-64=12 gxx_linux-64=12 git
PY=$MAMBA_ROOT_PREFIX/envs/hi3d/bin/python
$PY -m pip install -q torch==2.5.0+cu124 torchvision==0.20.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
$PY -m pip install -q packaging ninja psutil
$PY -m pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
(cd Hi3Dv1.0 && $PY -m pip install -q -c docker/constraints.txt -e ".[inference]")
$PY -m pip install -q "huggingface_hub[cli]"
HF=$MAMBA_ROOT_PREFIX/envs/hi3d/bin/huggingface-cli
mkdir -p weights
$HF download hi3d-official/Hi3Dv1.0 --revision d502e938cbedc6e87aba8884614c5ca6d177a4b6 --local-dir weights/Hi3Dv1.0 --exclude "assets/*"
$HF download facebook/dinov2-large --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c --local-dir weights/dinov2-large --include "*.json" "*.safetensors"
$HF download openai/clip-vit-large-patch14 --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 --local-dir weights/clip-vit-large-patch14 --include "*.json" "*.txt" "model.safetensors"
sha256sum weights/Hi3Dv1.0/*/model.safetensors
echo SETUP_DONE
# Run (from the engine repo mounted at /mnt/c/adbosp-vector3d-engine):
#   export CC=$MAMBA_ROOT_PREFIX/envs/hi3d/bin/x86_64-conda-linux-gnu-gcc
#   $PY vector3d/lab/backends/hi3d_runner.py --repo ~/ai3d/Hi3Dv1.0 --result R.json -- \
#     --model ~/ai3d/weights/Hi3Dv1.0 --local-files-only --dino-model ~/ai3d/weights/dinov2-large \
#     --clip-model ~/ai3d/weights/clip-vit-large-patch14 --image I.png --coarse-mesh C.glb --output O.glb \
#     --resolution 512 --seed 0 --sequential-cpu-offload
