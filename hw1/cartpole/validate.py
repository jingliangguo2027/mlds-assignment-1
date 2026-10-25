"""检查手算回报、梯度方向、测试时参数固定，以及与随机动作的表现对照。"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from train import discounted_returns, make_policy, evaluate


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(42)
    actual = discounted_returns([1, 1, 1])
    expected = torch.tensor([2.8525, 1.95, 1.0])
    assert torch.allclose(actual, expected, atol=1e-6)

    # 人工指定正回报只用于检查梯度方向，训练回报来自环境。
    check_policy = make_policy()
    state = torch.tensor([0.1, 0.0, 0.05, -0.02])
    before = check_policy(state)[0].item()
    optimizer = torch.optim.Adam(check_policy.parameters(), lr=0.001)
    loss = -10.0 * torch.log(check_policy(state)[0])
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in check_policy.parameters())
    optimizer.step()
    after = check_policy(state)[0].item()
    assert after > before

    folder = Path(__file__).resolve().parent / "results"
    policy = make_policy()
    policy.load_state_dict(torch.load(folder / "policy.pt", weights_only=True))
    saved_weights = {name: value.clone() for name, value in policy.state_dict().items()}
    env = gym.make("CartPole-v1")
    env.reset(seed=2026)
    evaluate(policy, env, episodes=2)
    env.close()
    assert all(torch.equal(saved_weights[name], value) for name, value in policy.state_dict().items())

    # 另做 100 局均匀随机动作，检查训练后的表现是否改善。
    env = gym.make("CartPole-v1")
    env.reset(seed=2026)
    env.action_space.seed(2026)
    random_scores = []
    for _ in range(100):
        env.reset()
        score = 0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            score += reward
            if terminated or truncated:
                break
        random_scores.append(score)
    env.close()
    results = json.loads((folder / "scores.json").read_text())
    assert len(results["test_scores"]) == 500
    assert results["test_mean"] > np.mean(random_scores)
    checks = {
        "hand_calculated_returns_passed": True,
        "positive_return_increases_selected_probability": [before, after],
        "evaluation_keeps_weights_fixed": True,
        "random_policy_episodes": 100,
        "random_policy_mean": float(np.mean(random_scores)),
        "trained_policy_episodes": 500,
        "trained_policy_mean": results["test_mean"],
    }
    (folder / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
