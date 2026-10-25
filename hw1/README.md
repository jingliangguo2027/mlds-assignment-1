# Assignment 1 code

All assignment code, outputs and reference PDFs are kept here. Original learning versions are retained.

## Shuttle dispatching

- `main.py`: finite horizon T=500, backward dynamic programming; `finite_horizon_value.png`.
- `value_iteration.py`: infinite-horizon value iteration; `value_iteration_value.png`.
- `policy_iteration/main.py`: policy evaluation/improvement; `policy_iteration/optimal_policy.png`.
- `policy_iteration/validate.py`: zero-discount experiment, analytical boundary checks and three-algorithm comparison. Use `--experiment all`.

Model: observe waiting passengers, choose whether to dispatch, charge dispatch plus remaining-passenger waiting cost, then add independent uniform arrivals 1–5. State range 0–200. Dispatch capacity 15, dispatch cost 100, waiting cost 2, gamma=0.95. Excess arrivals above 200 are excluded from the queue, a stated modeling assumption. The finite-horizon terminal cost is zero.

## Policy gradient

- `cartpole/train.py`: the original annotated teaching implementation; paired with `cartpole/逐行讲解.md`.
- `policy_gradient/`: complete CartPole/Pong runner with no-baseline and historical EMA variants, single-GPU launcher, validations, resume tests and result summarizer.

Read `policy_gradient/README.md` for setup and exact commands. PyTorch on Deepdish follows the Lab CUDA 12.6 installation command; the code records actual versions. Do not upload `.venv/`, passwords, SSH private keys or GitHub tokens.

`package_code.py` produces `hw1_code.zip`, a small code/documentation bundle for uploading to the cluster. Local environments, results and partially trained models are excluded. Run fresh experiments on the server, then resume only that server's own checkpoints.

## Completed report and results

The final report is `output/pdf/assignment1_report.pdf`. It includes all required plots, correctness and boundary checks, run commands, and the three-month validation discussion. Rebuild it with `python build_report.py` after installing ReportLab.

Repository: https://github.com/jingliangguo2027/mlds-assignment-1 (private). The instructor needs access before submission. Nothing has been submitted to Canvas.

All four frozen-policy tests contain 500 episodes. CartPole trained for 2000 episodes per setting; the selected Pong runs trained for 5000. `policy_gradient/report_results.md` contains the final summary. Tuning runs remain separate from the final tests. Large model files and local environments are not included in Git; recorded scores and model hashes are included.
