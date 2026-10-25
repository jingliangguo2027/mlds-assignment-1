"""Check both loss gradients analytically, including trajectories spanning chunks."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from train import discounted_returns, update
from validate import exact_gradient_check


def gradient_check():
    torch.manual_seed(19)
    x = torch.randn(600, 3)
    actions = torch.randint(2, (600,))
    rewards = np.zeros(600, dtype=np.float32)
    rewards[[5, 70, 280, 599]] = [1, -1, 1, -1]
    returns = discounted_returns(rewards, .99)
    # Inner discount remains intact when the actor's outer discount is removed.
    np.testing.assert_allclose(discounted_returns([1., 0., 2.], .99),
                               [1 + .99**2 * 2, .99 * 2, 2], rtol=1e-6)
    model = nn.Linear(3, 2)
    errors, gradients = {}, {}
    for omit in (False, True):
        for baseline in (0., .4):
            with torch.no_grad():
                probs = model(x).softmax(-1)
                one_hot = torch.nn.functional.one_hot(actions, 2)
                advantage = torch.tensor(returns) - baseline
                weights = torch.ones(600) if omit else .99**torch.arange(600)
                # Closed-form derivative of the softmax negative log likelihood.
                residual = (probs - one_hot) * (advantage * weights)[:, None]
                expected_weight = residual.T @ x
                expected_bias = residual.sum(0)
            update(model, torch.optim.SGD(model.parameters(), lr=0), x.numpy(),
                   actions.tolist(), returns, .99, baseline, 'cpu',
                   omit_time_discount=omit)
            torch.testing.assert_close(model.weight.grad, expected_weight, rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(model.bias.grad, expected_bias, rtol=1e-5, atol=1e-4)
            errors[f'omit={omit},baseline={baseline}'] = float((model.weight.grad-expected_weight).abs().max())
            gradients[(omit, baseline)] = model.weight.grad.clone()
    assert not torch.allclose(gradients[(False, 0.)], gradients[(True, 0.)])
    return errors


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = {'original_discounted_objective_errors': exact_gradient_check(),
              'analytical_softmax_gradient_errors': gradient_check(),
              'inner_discount_preserved': True, 'trajectory_length': 600}
    path = Path('tuning_results/loss_mode_checks.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
