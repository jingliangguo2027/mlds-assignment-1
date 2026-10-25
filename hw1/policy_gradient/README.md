# Policy gradient: complete runnable code

## Completed final results (October 25)

All four 500-episode tests are complete. See `report_results.md` and `../output/pdf/assignment1_report.pdf`. Pong test mean / standard deviation: no baseline -18.374 / 2.743; EMA -18.786 / 2.709. The final settings and reproduction commands in the PDF take precedence over the historical launchers described below. No further training is running.

## Selected final Pong tests

The selected no-baseline and historical-EMA policies both completed 5000 training episodes with 128 hidden units, `he_small_output` initialization, Adam lr=0.001, batch1, and the original inner/outer gamma=0.99 discounts. Their immutable source models are `tuning_results/width128_he_small/checkpoint_5000.pt` and `tuning_results/width128_he_small_ema/checkpoint_5000.pt`.

`evaluate_pong_final.py` performs only frozen stochastic-policy evaluation: 500 games per model on the reserved seeds1000000–1000499, sequentially on one GPU. It never updates policy parameters or baseline. `tuning_results/final_test_500.json` binds the selected model fingerprints and checkpoint hashes; a lock prevents duplicate evaluators, and saved test records resume every10 games. These final results use `results/pong_final_none/` and `results/pong_final_ema/`; historical results are retained. Do not start a second evaluator while one is running. `summarize.py --require-complete` includes these selected Pong final results and the existing matched CartPole experiments only after all four have complete500-game tests. Earlier50-game tuning results remain separate.

## Files to read

- `train.py`: one shared REINFORCE implementation for CartPole and Pong, optional baseline, checkpointing and testing.
- `validate.py`: analytical gradient proof by enumeration, Pong preprocessing and deterministic replay, and paired gradient variance.
- `test_resume.py`: short real runs for all four game/baseline combinations; continuation and stale-test checks.
- `summarize.py`: compares completed runs, draws baseline-comparison plots and prepares report tables.
- `deepdish_setup.sh`, `run_deepdish.sh`: Lab environment setup and single-GPU sequential execution.

The original beginner-friendly CartPole implementation remains in `../cartpole/train.py`.

## Methods

CartPole uses `CartPole-v1`, four raw state values, a 4–32–2 ReLU network, gamma=0.95 and Adam lr=0.01. Pong uses **`ALE/Pong-v5`**, the modern registered name of the requested v5 environment. It retains v5 defaults: RGB, frameskip=4, repeat_action_probability=0.25. The policy has two outputs mapped **only to actions 2 and 3**. No FIRE wrapper or alternate v0/v4 environment is used.

Pong preprocessing follows the assignment: crop rows 35:195, downsample by 2, remove backgrounds 144 and 109, and binarize. Consecutive processed frames are subtracted to expose motion; the 80×80 difference is flattened to 6400 inputs. The network is 6400–128–2 with ReLU. Pong gamma=0.99. The original historical run script uses Adam lr=0.0001, a documented deviation from the suggested 0.01 to use smaller policy updates. This is an implementation choice, not a demonstrated optimum.

The objective is full-episode discounted reward. Return `G_t = r_t + gamma*r_(t+1) + ...`; actor loss is `-sum(gamma**t * (G_t-b) * log pi(a_t|s_t))`. Returns are not reset after Pong points. Every episode is collected with a fixed policy, then the actor is updated once. Chunked gradient accumulation limits memory use while preserving the summed episode loss. This is basic REINFORCE; there is no PPO, DQN, critic or return normalization.

An earlier user-requested Pong comparison removes **only the outer** `gamma**t`: `--omit-time-discount` selects `-sum((G_t-b)*log pi)`, while `G_t` retains its internal discount. The new experiment `no_outer_lr001` starts from seed 0 with Adam lr=0.001, baseline=none and a first 300-episode checkpoint. It is compared with the existing `lr_001` run at the same training budget and validation seeds. This is a different gradient estimator, not claimed to be the exact gradient of the original full-episode discounted objective. Historical runs and default behavior are preserved; resuming a checkpoint with a different loss mode is rejected. `validate_loss_modes.py` checks the closed-form softmax gradient for both modes and both baseline values across 600 steps.

The completed 300-episode architecture comparison restored the outer discount and fixes lr=0.001. It compares hidden widths 32/64/128/256 with default initialization, reusing the original `lr_001/checkpoint_300.pt` as the unchanged 128-unit control. A separate 128-unit `he_small_output` run compares initialization: He-normal ReLU weights, zero biases and an orthogonal output layer with gain 0.01. Each new run starts fresh and trains exactly 300 episodes. `run_architecture_sweep.py` executes the four new runs sequentially on one GPU; `tuning_results/architecture_sweep.json` records their bounded budgets and status. Each architecture/initialization has its own untrained validation reference. All trained models share 50 tuning-validation seeds; these are not final tests. `--hidden` and `--initialization` are supported by both training entry points.

For `--baseline none`, b=0. For `--baseline ema`, b is a scalar historical reference. Initialize b=0; use that old b for the entire episode's update. **After** updating the actor, compute `reference = sum(gamma**t * G_t) / sum(gamma**t)` and update `b = 0.9*b + 0.1*reference`. Thus the baseline is based only on preceding episodes when used; it does not depend on the currently sampled action.

Training and evaluation reset the environment with distinct, recorded per-episode seeds. All final evaluations use the frozen stochastic policy for 500 full episodes. Episode scores plotted in figures are undiscounted sums. Test standard deviation uses ddof=0. Both terminal and time-limit flags end the task episode; no value bootstrapping is used.

## Local code checks

From this folder, using the already prepared local environment:

```sh
../cartpole/.venv/bin/python validate.py
../cartpole/.venv/bin/python test_resume.py
../cartpole/.venv/bin/python summarize.py
```

For a fresh CPU environment, install PyTorch from its official distribution, then `python -m pip install -r requirements.txt`. The local environment currently also contains `ale-py==0.12.1`, which supplies the ALE integration used by the tests.

Example short trial (not an assignment-length evaluation):

```sh
python train.py --game pong --baseline ema --episodes 1 --eval-episodes 2 --lr 0.0001 --output smoke/example
```

## Running on Deepdish

Follow `MLDS_Lab_Week_1.pdf`: campus network or Northwestern VPN, SSH with NetID, HW1 folder, GitHub setup, conda environment named `myenv`, CUDA PyTorch and `torch.cuda.is_available()` returning True. The setup utility's current official instructions are at https://github.com/NUMLDS/mlds-setup-releases. Do not send passwords in chat or put credentials into this repository.

For a fresh server run, use `../package_code.py` to create `../hw1_code.zip`. This code-only archive excludes local environments, all local results and checkpoints, so a new GPU run cannot silently continue CPU trials and then be mislabeled as GPU training. Upload and extract that archive in your server folder. From `policy_gradient/`:

```sh
bash deepdish_setup.sh
nvidia-smi
# Choose ONE available GPU. The number 0 below is an example, not a GPU reservation.
CUDA_VISIBLE_DEVICES=0 nohup bash run_deepdish.sh > training.log 2>&1 < /dev/null &
tail -f training.log
```

The launcher runs four experiments sequentially on that single visible GPU: CartPole none/EMA, each 2000 training episodes; Pong none/EMA, each 3000 training episodes. Each is followed by **500** test episodes. Jobs continue after SSH disconnect. The code rejects CUDA execution if more than one GPU is visible. No server job has been claimed as completed merely because local CPU runs exist.

For the original 3000-episode Pong setup, use `run_pong_deepdish.sh` instead. It writes `logs/PONG_COMPLETE` only after both variants finish 3000 training and 500 evaluation episodes. The launch scripts also locate Deepdish's `/opt/anaconda/bin/conda` in noninteractive SSH sessions.

## Results and continuation

### Small Pong tuning experiments (October 24)

`tune_pong.py` records which loss mode is used; historical runs retain the outer discount, while `no_outer_lr001` explicitly omits it. The current default is at most 300 additional training episodes per invocation, adjustable with `--max-block`. Fresh candidates start from seed 0 and share training/validation seeds. It records gradients, parameter changes and action entropy; the first screening used 20 validation episodes and later stages use 50. These are separate from the final 500 test episodes. `validate_tuning.py` verifies the original analytical gradient and batch averaging; `validate_loss_modes.py` verifies both loss modes; `diagnose_pong_variance.py` measures paired Pong gradient variance. `staged_report.py` rebuilds the current report and plots; `summarize_tuning.py` belongs to the earlier 100-episode screening.

`fork_tuning.py` can create a distinctly named learning-rate branch from a completed stage. It preserves model parameters, Adam moments and episode numbering, changes only the learning rate, and refuses to overwrite an existing folder. `lineage.json` identifies the source checkpoint and hash; `validation_at_fork.json` is the source model's existing validation, not a newly trained result. The branch can then continue with `tune_pong.py` and the same future training seeds as its parent.

The first five 100-episode candidates all failed the initial continuation threshold. That hard stopping rule was subsequently retired: the current protocol usually checks every 300 episodes, expands tuning validation to 50 episodes, and considers score, discounted objective, exploration and multiple-stage trends. Existing 100-episode checkpoints are retained; the first extended check is at a total of 300 episodes. `staged_report.py` builds `tuning_results/分阶段实验分析.md` from recorded results and decisions; the earlier `首轮100局实验分析.md` is an archived report, not current job status.

Each `results/<game>_<baseline>/` contains:

- `config.json`: parameters, preprocessing, device, library versions.
- `checkpoint.pt`: actor, Adam state, baseline value, training history, CPU/CUDA random states.
- `progress.json` and `status.json`: current progress; status is training/evaluating/complete.
- `first_10_steps.json`: actual early actions, probabilities, rewards, terminal flags.
- `evaluation_scores.json`: test scores bound to a fingerprint of the exact model.
- `metrics.json`, `training_rewards.png`, `evaluation_rewards.png`: final scores, 100-episode moving average and reward histogram.

Checkpoints are saved every 10 training episodes and at the final episode. Restart with exactly the same game, baseline, lr and seed, and add `--resume`. You may increase the episode budget. Up to nine unsaved episodes can be repeated after interruption. Test scores resume every 10 episodes, with per-episode seeds for reproducibility. Training more invalidates scores from the old model automatically. Fresh runs cannot silently overwrite an existing checkpoint.

Before final reporting:

```sh
python summarize.py --require-complete
```

This fails if any of the four experiments is missing a complete 500-episode evaluation. Without this flag it produces an explicitly incomplete table. Do not report smoke runs or intermediate Pong scores as final results.

## Validation evidence and limits

The two-step analytical toy problem enumerates all four action sequences and compares the probability-weighted REINFORCE gradient to direct differentiation of its known expected objective. It checks both b=0 and b=0.8, including the outer gamma**t factor and baseline invariance.

The variance experiment first estimates a historical EMA reference using 20 separate pilot episodes, then measures full policy-parameter gradients on 40 new trajectories under the same fixed policy. Each new trajectory is used both with and without the frozen reference. The statistic is the sum of per-parameter sample gradient variances. The measured ratio is reported, not assumed to be below one. This is a variance diagnostic, not proof of better results across training seeds.

Pong validation checks binary preprocessing, input immutability, allowed action mapping and exact replay under fixed environment/action seeds, including a complete actual episode. The continuation test compares resumed and uninterrupted training and catches stale evaluations after further training. Future three-month work should add multiple training seeds with confidence intervals, matched learning-rate sweeps, more baseline choices, out-of-distribution starts and longer Pong budgets.

The old teaching CartPole run uses a different episode-reset protocol and is kept separately. Its reported 498.552 mean is **not** substituted for the matched no-baseline run when comparing against the new EMA implementation.
