"""Analytical gradient check, Pong preprocessing/actions, and paired gradient variance."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from train import discounted_returns, make_env, make_policy, preprocess, rollout, update


def exact_gradient_check():
    torch.manual_seed(2)
    model = nn.Linear(1, 2, bias=False)
    probabilities = model(torch.ones(1)).softmax(-1)
    objective = probabilities[0] + 0.95 * 2 * probabilities[1]
    exact = torch.autograd.grad(objective, model.weight)[0].detach()
    p = probabilities.detach()
    errors = {}
    for baseline in (0.0, 0.8):
        average = torch.zeros_like(exact)
        for a0, a1 in itertools.product((0, 1), repeat=2):
            rewards = [float(a0 == 0), 2.0 * (a1 == 1)]
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            update(model, optimizer, np.ones((2, 1), dtype=np.float32), [a0, a1],
                   discounted_returns(rewards, 0.95), 0.95, baseline, "cpu")

            average -= p[a0] * p[a1] * model.weight.grad
        error = float((average-exact).abs().max())
        assert error < 1e-6
        errors[str(baseline)] = error
    return errors


def pong_checks():
    image = np.full((210, 160, 3), 144, dtype=np.uint8)
    image[35, 0, 0] = 109
    image[37, 2, 0] = 255
    original = image.copy()
    reduced = preprocess(image)
    assert reduced.shape == (80, 80) and reduced[0, 0] == 0 and reduced[1, 1] == 1
    assert np.array_equal(image, original)
    assert set(np.unique(reduced)) <= {0, 1}
    policy = make_policy("pong")
    env = make_env("pong")
    torch.manual_seed(77)
    first = rollout(env, policy, "pong", "cpu", 77)
    torch.manual_seed(77)
    second = rollout(env, policy, "pong", "cpu", 77)
    env.close()
    assert first[1] == second[1] and first[2] == second[2]
    assert all(row["environment_action"] in (2, 3) for row in first[3])
    assert first[2].count(-1.0) + first[2].count(1.0) > 0
    return {"preprocessing_passed": True, "seeded_replay_identical": True,
            "episode_score": sum(first[2]), "episode_steps": len(first[1]), "first_10_steps": first[3]}


def gradient_variance():
    torch.manual_seed(91)
    policy = make_policy("cartpole")
    env = make_env("cartpole")
    baseline = 0.0
    for episode in range(20):
        _, _, rewards, _ = rollout(env, policy, "cartpole", "cpu", 5000+episode)
        returns = discounted_returns(rewards, 0.95)
        reference = float(np.average(returns, weights=0.95**np.arange(len(rewards))))
        baseline = 0.9 * baseline + 0.1 * reference
    samples = {"none": [], "historical_ema": []}
    for episode in range(40):
        states, actions, rewards, _ = rollout(env, policy, "cartpole", "cpu", 6000+episode)
        returns = torch.tensor(discounted_returns(rewards, 0.95))
        weights = 0.95 ** torch.arange(len(actions))
        for name, b in (("none", 0.0), ("historical_ema", baseline)):
            policy.zero_grad()
            log_probs = Categorical(logits=policy(torch.tensor(states))).log_prob(torch.tensor(actions))
            loss = -(weights * (returns-b) * log_probs).sum()
            loss.backward()
            samples[name].append(torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().clone())
    env.close()
    variances = {name: float(torch.stack(rows).var(dim=0, unbiased=True).sum()) for name, rows in samples.items()}
    return {"pilot_episodes": 20, "paired_episodes": 40, "baseline": baseline,
            "sum_of_component_gradient_variances": variances,
            "variance_ratio_ema_to_none": variances["historical_ema"]/variances["none"]}


def pong_variance(device):
    torch.manual_seed(0)
    policy = make_policy('pong').to(device)
    env = make_env('pong')
    baseline = 0.0
    for seed in range(300000, 300020):
        torch.manual_seed(seed)
        _, _, rewards, _ = rollout(env, policy, 'pong', device, seed)
        returns = discounted_returns(rewards, .99)
        baseline = .9 * baseline + .1 * float(np.average(returns, weights=.99**np.arange(len(returns))))
    samples = {'none': [], 'ema': []}
    for seed in range(310000, 310020):
        torch.manual_seed(seed)
        states, actions, rewards, _ = rollout(env, policy, 'pong', device, seed)
        for name, b in [('none', 0), ('ema', baseline)]:
            update(policy, torch.optim.SGD(policy.parameters(), lr=0), states, actions,
                   discounted_returns(rewards, .99), .99, b, device)
            samples[name].append(torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().cpu())
    env.close()
    variances = {name: float(torch.stack(rows).var(0, unbiased=True).sum()) for name, rows in samples.items()}
    return dict(baseline=baseline, variances=variances, ratio=variances['ema']/variances['none'])


def chunk_check(device):
    torch.manual_seed(0)
    policy = make_policy('pong').to(device)
    env = make_env('pong')
    torch.manual_seed(123)
    states, actions, rewards, _ = rollout(env, policy, 'pong', device, 123)
    env.close()
    returns = discounted_returns(rewards, .99)
    update(policy, torch.optim.SGD(policy.parameters(), lr=0), states, actions, returns, .99, 0, device)
    chunked = torch.cat([p.grad.flatten() for p in policy.parameters()]).clone()
    policy.zero_grad()
    x = torch.tensor(states, dtype=torch.float32, device=device)
    a = torch.tensor(actions, device=device)
    weights = torch.tensor(returns, device=device) * .99**torch.arange(len(actions), device=device)
    loss = -(Categorical(logits=policy(x)).log_prob(a) * weights).sum()
    loss.backward()
    full = torch.cat([p.grad.flatten() for p in policy.parameters()])
    error = float((chunked-full).norm()/full.norm())
    assert error < 1e-4
    with torch.no_grad():
        for p in policy.parameters():
            p -= 1e-5*p.grad/full.norm()
        after = float(-(Categorical(logits=policy(x)).log_prob(a)*weights).sum())
    assert after < float(loss.detach())
    return dict(relative_error=error, loss_before=float(loss.detach()), loss_after=after)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', choices=['pong-variance', 'chunk'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    if args.extra:
        result = pong_variance(args.device) if args.extra == 'pong-variance' else chunk_check(args.device)
    else:
        result = {'exact_gradient_max_errors': exact_gradient_check(),
                  'pong': pong_checks(), 'gradient_variance': gradient_variance()}
    # Print new results; never overwrite the submitted evidence.
    print(json.dumps(result, indent=2))
