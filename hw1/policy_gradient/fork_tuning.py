"""Continue a completed tuning checkpoint with a new learning rate only."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
from train import load_checkpoint, save_json


def fork(source, destination, lr):
    checkpoint = load_checkpoint(source)
    episodes = len(checkpoint['records'])
    result_path = source.parent / f'block_{episodes}.json'
    result = json.loads(result_path.read_text())
    assert result['episodes'] == episodes
    assert lr > 0
    # A distinct, new directory protects both the parent and earlier branches.
    destination.mkdir(parents=True, exist_ok=False)
    previous_lr = checkpoint['config']['lr']
    checkpoint['config']['lr'] = lr
    for group in checkpoint['optimizer']['param_groups']:
        group['lr'] = lr
    torch.save(checkpoint, destination / 'checkpoint.pt')
    torch.save(checkpoint, destination / f'checkpoint_{episodes}.pt')
    save_json(destination / 'config.json', checkpoint['config'])
    save_json(destination / 'validation_at_fork.json', result)
    save_json(destination / 'lineage.json', {
        'source_checkpoint': str(source), 'source_experiment': source.parent.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'episodes_at_fork': episodes, 'previous_lr': previous_lr, 'new_lr': lr,
        'optimizer_moments_preserved': True,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    })
    print(f'{destination}: inherited {episodes} episodes; lr {previous_lr} -> {lr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--lr', type=float, required=True)
    args = parser.parse_args()
    fork(args.source, args.destination, args.lr)
