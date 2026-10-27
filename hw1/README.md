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
| `policy_gradient/pong_standardized.py` | Separate Pong follow-up: standardized full-game returns and plots |
| `policy_gradient/results/` | Original tests, follow-up training curves and frozen-model tests |
| `policy_gradient/evidence.json` | Saved validation and selected tuning evidence |
| `build_report.py` | Verify scores, redraw report figures and rebuild the 8-page report |

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

Checkpoints are saved every 10 episodes. Add `--resume` with the same settings to continue a checkpoint created by this runner. Evaluation reruns from fixed test seeds. Older checkpoint formats use the corresponding code in Git history.

## Validation and report

The [report](output/pdf/assignment1_report.pdf) retains the original baseline comparison and adds the standardized-return Pong experiment. The last 100 games in the 4000-game training window average **+7.20 with 89% wins**. Training stopped at 4160. Its frozen 500-game test scores **+5.938 (SD 5.859), with 85% wins (425/500)**. Full results are in `results/pong_standardized/evaluation.json`.

### Pong follow-up

This separate experiment keeps full-game returns with gamma=0.99, standardizes advantages within each episode, and removes the outer `gamma**t` weight. It uses Adam lr=0.001, 10 episodes per update, 128 hidden units and output gain=0.1. A constant EMA cancels during centering. Both update changes were applied together, so this is not an isolated test of standardization.

From `policy_gradient`, use a new folder to reproduce the method:

```sh
CUDA_VISIBLE_DEVICES=0 python pong_standardized.py --episodes 4160 --device cuda --output new_pong_standardized
python pong_standardized.py --plot-only --output results/pong_standardized
```

The first command trains a new model. Raw training scores, settings, milestones and the source checkpoint hash are in `results/pong_standardized/metrics.json`.

The command above trains to the final 4160-game endpoint. Test the saved model without training:

```sh
CUDA_VISIBLE_DEVICES=0 python pong_standardized.py --evaluate-checkpoint new_pong_standardized/checkpoint.pt --eval-episodes 500 --vector-envs 8 --device cuda --output new_pong_test
```

This evaluator uses 500 new environment seeds, 9000000–9000499, and independent per-game action RNGs. It samples the frozen policy and verifies that weights and the checkpoint do not change. Parallel CPU environments share one GPU. The 4000-game charts are training statistics, separate from the test of the 4160-game checkpoint.

### Checks and rebuilding

```sh
# From policy_gradient
python validate.py
CUDA_VISIBLE_DEVICES=0 python validate.py --extra pong-variance --device cuda
CUDA_VISIBLE_DEVICES=0 python validate.py --extra chunk --device cuda
# From hw1
python build_report.py
```

The default checks compare REINFORCE with an exact gradient, replay a seeded Pong trajectory, verify image processing, and measure paired CartPole gradient variance. Extra modes reproduce the Pong variance diagnostic and compare chunked/full gradients plus a descent step. They print new results rather than overwrite the submitted evidence.

The four original final tests each contain 500 games. Neither original Pong model wins a test game. The follow-up separates the 4000-game training window from the independent test at 4160 games. All settings use one training seed. Original tuning and validation scripts remain in [commit e8ec206](https://github.com/jingliangguo2027/mlds-assignment-1/tree/e8ec206580c0c26dac98153dbd1bf1438fd36f4b/hw1). Original scores and plots are preserved.

## Assignment coverage

| Requirement | Report / code |
|---|---|
| Shuttle enumeration (T=500), value iteration and policy iteration; gamma=0.95 | Sections 1–2; `main.py`, `value_iteration.py`, `policy_iteration/main.py` |
| CartPole-v1 and Pong-v5; specified discounts, Adam, Pong actions 2/3 | Section 3; `policy_gradient/train.py` |
| Explain a baseline and repeat the comparison | Sections 3–5; matched original none/EMA runs |
| Training curves with 100-game means; 500-game histograms, means and SDs | Sections 4–7; recorded data under `policy_gradient/results/` |
| Edge cases and quantitative validation beyond unit checks | Sections 1 and 7; both `validate.py` entry points |
| Additional experiments with three more months | Section 7 |
| GitHub identity, repository link and execution instructions | Section 8 and this README |
