"""Frozen 500-game final tests of the two selected 5000-episode Pong policies."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import numpy as np
import torch
from train import (load_checkpoint, make_policy, make_env, rollout, discounted_returns,
                   policy_fingerprint, save_json, plot_results)

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(1000000, 1000500))
RUNS = [('none', 'width128_he_small'), ('ema', 'width128_he_small_ema')]


def main():
    os.chdir(ROOT)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    lock = open('tuning_results/final_test_500.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = Path('tuning_results/final_test_500.json')
    plan = json.loads(plan_path.read_text())
    plan.update(status='running', pid=os.getpid())
    save_json(plan_path, plan)
    for baseline, name in RUNS:
        source = Path('tuning_results') / name / 'checkpoint_5000.pt'
        checkpoint = load_checkpoint(source)
        expected = plan['models'][baseline]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected['checkpoint_sha256']
        assert len(checkpoint['records']) == 5000
        assert checkpoint['config']['baseline'] == baseline
        policy = make_policy('pong', 128, 'he_small_output').to('cuda')
        policy.load_state_dict(checkpoint['policy'])
        policy.eval()
        policy.requires_grad_(False)
        model_id = policy_fingerprint(policy)
        assert model_id == expected['model_id']
        output = Path('results') / f'pong_final_{baseline}'
        output.mkdir(parents=True, exist_ok=True)
        config = dict(checkpoint['config'], game='pong', episodes=5000, eval_episodes=500,
                      device='cuda', environment='ALE/Pong-v5', evaluation_mode='frozen stochastic policy',
                      test_seed_start=SEEDS[0], test_seed_end=SEEDS[-1], source_checkpoint=str(source),
                      model_id=model_id, checkpoint_sha256=expected['checkpoint_sha256'], std_ddof=0)
        if (output / 'config.json').exists():
            assert json.loads((output / 'config.json').read_text()) == config
        save_json(output / 'config.json', config)
        scores_path = output / 'evaluation_scores.json'
        rows = []
        if scores_path.exists():
            saved = json.loads(scores_path.read_text())
            assert saved['model_id'] == model_id and saved['purpose'] == 'final_test'
            rows = saved['rows']
            assert [r['seed'] for r in rows] == SEEDS[:len(rows)] and len(rows) <= 500
            assert saved['scores'] == [r['score'] for r in rows]
        save_json(output / 'status.json', dict(status='evaluating', model_id=model_id,
                  training_episodes=5000, test_episodes=len(rows), target_test_episodes=500))
        plan.update(current=baseline, current_test_episodes=len(rows))
        save_json(plan_path, plan)
        env = make_env('pong')
        with torch.no_grad():
            for seed in SEEDS[len(rows):]:
                torch.manual_seed(seed)
                _, _, rewards, _ = rollout(env, policy, 'pong', 'cuda', seed)
                rows.append(dict(seed=seed, score=float(sum(rewards)), steps=len(rewards),
                                 discounted_return=float(discounted_returns(rewards, .99)[0])))
                if len(rows) % 10 == 0:
                    assert policy_fingerprint(policy) == model_id
                    save_json(scores_path, dict(model_id=model_id, purpose='final_test',
                              rows=rows, scores=[r['score'] for r in rows]))
                    save_json(output / 'status.json', dict(status='evaluating', model_id=model_id,
                              training_episodes=5000, test_episodes=len(rows), target_test_episodes=500))
                    plan.update(current_test_episodes=len(rows), last_checked_utc=datetime.now(timezone.utc).isoformat())
                    save_json(plan_path, plan)
                    print(f'{baseline}: final test {len(rows)}/500', flush=True)
        env.close()
        assert policy_fingerprint(policy) == model_id
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected['checkpoint_sha256']
        scores = [r['score'] for r in rows]
        history = [r['score'] for r in checkpoint['records']]
        metrics = dict(config=config, training_scores=history, test_scores=scores,
                       test_mean=float(np.mean(scores)), test_std=float(np.std(scores, ddof=0)),
                       test_episodes=500, training_episodes=5000, test_wins=sum(s > 0 for s in scores),
                       test_mean_discounted_return=float(np.mean([r['discounted_return'] for r in rows])),
                       model_id=model_id, purpose='final_test')
        save_json(output / 'metrics.json', metrics)
        plot_results(history, scores, output, 'pong', baseline)
        save_json(output / 'status.json', dict(status='complete', model_id=model_id,
                  training_episodes=5000, test_episodes=500, completed_utc=datetime.now(timezone.utc).isoformat()))
        plan.setdefault('completed_models', {})[baseline] = dict(mean=metrics['test_mean'], std=metrics['test_std'], output=str(output))
        save_json(plan_path, plan)
        print(json.dumps(plan['completed_models'][baseline]), flush=True)
    plan.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    save_json(plan_path, plan)


if __name__ == '__main__':
    main()
