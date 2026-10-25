#!/usr/bin/env bash
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  export PATH="/opt/anaconda/bin:$PATH"
fi
: "${CUDA_VISIBLE_DEVICES:?Set ONE available GPU index before starting}"
eval "$(conda shell.bash hook)"
conda activate myenv
task_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$task_dir"
mkdir -p logs
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print("GPU:", torch.cuda.get_device_name(0)); print("Torch:", torch.__version__)'
date -u
hostname
nvidia-smi
for baseline in none ema; do
  python -u train.py --game pong --baseline "$baseline" --episodes 3000 --eval-episodes 500 --lr 0.0001 --seed 0 --device cuda --output "results/pong_${baseline}" --resume
done
python - <<'PY'
import json
from pathlib import Path
for baseline in ('none', 'ema'):
    folder = Path('results') / ('pong_' + baseline)
    status = json.loads((folder/'status.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    assert status['status'] == 'complete'
    assert metrics['training_episodes'] == 3000
    assert metrics['test_episodes'] == len(metrics['test_scores']) == 500
    assert metrics['config']['device'] == 'cuda'
    print(baseline, 'mean:', metrics['test_mean'], 'std:', metrics['test_std'])
Path('logs/PONG_COMPLETE').write_text('Both Pong variants: 3000 training and 500 evaluation episodes each.\n')
PY
