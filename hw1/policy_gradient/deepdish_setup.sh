#!/usr/bin/env bash
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  export PATH="/opt/anaconda/bin:$PATH"
fi
# 先按 Lab 使用 myenv；已存在的环境不删除、不重建。
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx myenv; then
  conda create -n myenv python=3.12 pip -y
fi
conda activate myenv
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$(dirname "$0")/requirements.txt"
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available()'
