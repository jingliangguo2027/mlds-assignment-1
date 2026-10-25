# MLDS Assignment 1

GitHub username: jingliangguo2027

[Read the complete report (PDF)](hw1/output/pdf/assignment1_report.pdf)

The report uses simple English and includes the shuttle MDP, three solution methods, policy-gradient experiments, correctness checks, and a three-month validation plan. [Run instructions and code map](hw1/README.md).

## Final tests

Each setting uses 500 frozen-policy test episodes. Standard deviations use ddof=0.

| Game | Baseline | Training episodes | Test mean | Test standard deviation |
|---|---|---:|---:|---:|
| CartPole | None | 2000 | 123.094 | 36.230 |
| CartPole | Historical EMA | 2000 | 369.722 | 65.631 |
| Pong | None | 5000 | -18.374 | 2.743 |
| Pong | Historical EMA | 5000 | -18.786 | 2.709 |

Raw scores and figures are in `hw1/policy_gradient/results/`. Historical tuning results are separate in `tuning_results/`. These experiments use one training seed; neither final Pong policy wins a test game.

## Main code

- Shuttle: `hw1/main.py`, `hw1/value_iteration.py`, `hw1/policy_iteration/main.py`.
- Games: `hw1/policy_gradient/train.py`.
- Checks: `hw1/policy_iteration/validate.py`, `hw1/policy_gradient/validate.py`.
- Rebuild the report: from `hw1`, install ReportLab and run `python build_report.py`.

The extra scripts preserve tuning and validation history. Start with the main code above. Model checkpoints and local Python environments are not stored in Git; final model hashes and full recorded scores are included. The repository is private; grant the instructor access before submission.
