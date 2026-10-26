# assignment1

Jingliang Guo · NetID: nxt1510

## Where to start

| File | Purpose |
|---|---|
| `main.py` | Finite-horizon shuttle solution, T=500 |
| `value_iteration.py` | Infinite-horizon shuttle value iteration |
| `policy_iteration/main.py` | Shuttle policy iteration |
| `policy_gradient/train.py` | CartPole and Pong REINFORCE, with optional EMA baseline |
| `policy_iteration/validate.py` | Shuttle analytic and algorithm checks |
| `policy_gradient/validate.py` | Gradient, replay and variance checks |
| `policy_gradient/results/` | Four final experiments: scores/settings and required plots |
| `policy_gradient/evidence.json` | Saved validation and selected tuning evidence |
| `build_report.py` | Verify recorded score summaries and rebuild the report |

## Setup and shuttle

Install PyTorch for your CPU/CUDA setup, then run from `hw1`:

```sh
python -m pip install -r requirements.txt
python main.py
python value_iteration.py
python policy_iteration/main.py
python policy_iteration/validate.py --experiment all
```

## Games

Run each command twice, replacing `BASELINE` with `none` and `ema`. Use new output folders so submitted results stay unchanged.

```sh
cd policy_gradient
python train.py --game cartpole --baseline BASELINE --episodes 2000 --eval-episodes 500 --lr 0.01 --output new_cartpole_BASELINE
CUDA_VISIBLE_DEVICES=0 python train.py --game pong --baseline BASELINE --episodes 5000 --eval-episodes 500 --lr 0.001 --hidden 128 --initialization he_small_output --device cuda --output new_pong_BASELINE
```

The loss is `-sum(gamma**t * (G_t - b) * log_prob)`. Gamma is 0.95 for CartPole and 0.99 for Pong. Pong uses only actions 2/3, frame differences and the v5 environment defaults. The EMA baseline is updated after the policy, using preceding episodes when applied. Training uses one Adam step per episode.

Checkpoints are saved every 10 episodes. Add `--resume` with the same settings to continue a checkpoint created by this simplified runner. Evaluation reruns from the fixed test seeds. Historical checkpoints have a different format; use the historical code to load those.

## Validation and report

```sh
# From policy_gradient
python validate.py
CUDA_VISIBLE_DEVICES=0 python validate.py --extra pong-variance --device cuda
CUDA_VISIBLE_DEVICES=0 python validate.py --extra chunk --device cuda
# From hw1
python build_report.py
```

The default checks compare REINFORCE with an exact gradient, replay a seeded Pong trajectory, verify image processing, and measure paired CartPole gradient variance. Extra modes reproduce the Pong variance diagnostic and compare chunked/full gradients plus a descent step. They print new results rather than overwrite the submitted evidence.

All four saved final tests contain 500 games. These runs use one training seed; neither Pong model wins a final test game. Original tuning and validation scripts remain in [commit e8ec206](https://github.com/jingliangguo2027/mlds-assignment-1/tree/e8ec206580c0c26dac98153dbd1bf1438fd36f4b/hw1). Historical scripts are not needed for the current main workflow. The original scores are preserved; cleanup does not represent new training.
