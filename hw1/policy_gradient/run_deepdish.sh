#!/usr/bin/env bash
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  export PATH="/opt/anaconda/bin:$PATH"
fi
# 启动前先 nvidia-smi 检查占用，再指定一个空闲 GPU。
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to ONE available GPU index first}"
eval "$(conda shell.bash hook)"
conda activate myenv
task_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$task_dir"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
# 同一张 GPU 串行运行；none 与 ema 使用相同预算、学习率和随机种子。
for baseline in none ema; do
  python -u train.py --game cartpole --baseline "$baseline" --episodes 2000 --eval-episodes 500 --lr 0.01 --device cuda --output "results/cartpole_${baseline}" --resume
done
for baseline in none ema; do
  python -u train.py --game pong --baseline "$baseline" --episodes 3000 --eval-episodes 500 --lr 0.0001 --device cuda --output "results/pong_${baseline}" --resume
done
