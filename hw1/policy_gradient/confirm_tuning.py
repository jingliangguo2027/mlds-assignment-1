"""Frozen-policy confirmation on seeds not used by tuning or final testing."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import torch
from train import load_checkpoint, make_policy, policy_fingerprint, save_json
from tune_pong import evaluate, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed-start', type=int, default=300000)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh output path; preserve earlier evidence'
    assert 300000 <= args.seed_start < args.seed_start + args.episodes < 1000000
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = make_policy('pong').to('cuda')
    print('Validating untrained reference', flush=True)
    reference = evaluate(policy, args.episodes, args.seed_start)
    results = {'seed_start': args.seed_start, 'episodes': args.episodes,
               'status': 'running', 'requested_checkpoints': [str(p) for p in args.checkpoints],
               'purpose': 'fresh confirmation; separate from the final 500-episode test',
               'reference_config': {'hidden':128,'initialization':'default','seed':0},
               'reference': reference, 'candidates': []}
    for path in args.checkpoints:
        checkpoint = load_checkpoint(path)
        config = checkpoint['config']
        policy = make_policy('pong',config.get('hidden',128),config.get('initialization','default')).to('cuda')
        policy.load_state_dict(checkpoint['policy'])
        print(f'Validating {path}', flush=True)
        rows = evaluate(policy, args.episodes, args.seed_start)
        result = {'checkpoint': str(path), 'model_fingerprint': policy_fingerprint(policy),
                  'training_episodes': len(checkpoint['records']), 'config': checkpoint['config'],
                  'validation': rows, 'vs_initial': comparison(rows, reference)}
        if results['candidates']:
            result['vs_first_candidate'] = comparison(rows, results['candidates'][0]['validation'])
        results['candidates'].append(result)
        save_json(args.output, results)
        print(json.dumps({k: v for k, v in result.items() if k not in ('validation', 'config')}, indent=2), flush=True)
    results['status'] = 'complete'
    results['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, results)
    print('CONFIRMATION_COMPLETE', flush=True)
