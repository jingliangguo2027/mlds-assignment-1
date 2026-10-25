"""Build the report from saved results. This script does not train any model."""
from pathlib import Path
import json
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Preformatted

ROOT = Path(__file__).resolve().parent
PG = ROOT / 'policy_gradient'
OUT = ROOT / 'output/pdf/assignment1_report.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
WIDTH = A4[0] - 88
BLUE = colors.HexColor('#173957')
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=BLUE, spaceAfter=12),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=BLUE, spaceBefore=9, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.2, leading=14.3, spaceAfter=7),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.5, leading=11.5, spaceAfter=6),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.7, leading=11.8),
    'code': ParagraphStyle('code', fontName='Courier', fontSize=8, leading=11, spaceAfter=9),
}
story = []


def tidy(text):
    # Add spaces in prose such as "of500"; keep paths and the GitHub username.
    return re.sub(r'(?<![\w/])([A-Za-z]{2,})(?=\d)',
                  lambda m: m[1] if m[1] in {'jingliangguo', 'hw'} else m[1] + ' ', str(text))


def read(path):
    return json.loads((ROOT / path).read_text())


def p(text, style='body'):
    story.append(Paragraph(tidy(text), styles[style]))


def page(title):
    if story:
        story.append(PageBreak())
    p(title, 'title')


def code(text):
    story.append(Preformatted(text, styles['code']))


def table(rows, widths=None):
    widths = widths or [1] * len(rows[0])
    cells = [[Paragraph(escape(tidy(x)), styles['cell']) for x in row] for row in rows]
    t = Table(cells, colWidths=[WIDTH*w/sum(widths) for w in widths], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4edf4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), .8, BLUE),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#cbd3da')),
    ]))
    story.extend([t, Spacer(1, 8)])


def figure(path, caption, height=266):
    img = Image(str(ROOT / path))
    scale = min(WIDTH/img.imageWidth, height/img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.extend([img, Spacer(1, 3)])
    p(caption, 'small')


metrics = {name: read(f'policy_gradient/results/{name}/metrics.json') for name in
           ['cartpole_none', 'cartpole_ema', 'pong_final_none', 'pong_final_ema']}
for m in metrics.values():
    assert len(m['test_scores']) == 500
comparison = read('policy_gradient/final_test_comparison.json')
checks = read('policy_gradient/validation_results.json')
shuttle = read('policy_iteration/validation_comparison.json')

page('Assignment 1\nReinforcement Learning')
p('MLDS 490 / IEMS 469 | October 25, 2026', 'heading')
table([
    ['Submission information', 'Value'],
    ['GitHub username', 'jingliangguo2027'],
    ['Repository', 'mlds-assignment-1'],
    ['Repository URL', 'https://github.com/jingliangguo2027/mlds-assignment-1'],
], [1, 2.3])
p('Code and results: <link href="https://github.com/jingliangguo2027/mlds-assignment-1" color="#173957">github.com/jingliangguo2027/mlds-assignment-1</link>. The repository is private. The instructor must have access before submission.')
p('1. Summary', 'heading')
p('I solved the shuttle problem with finite-horizon dynamic programming, value iteration, and policy iteration. I also used basic REINFORCE to train CartPole and Pong, both with and without a baseline. All results below come from saved runs. Failed or weak results are kept.')
rows = [['Game', 'Baseline', 'Training games', 'Test games', 'Mean score', 'Std. dev.']]
for name, m in metrics.items():
    rows.append(['CartPole' if name.startswith('cartpole') else 'Pong', m['config']['baseline'],
                 len(m['training_scores']), 500, f'{m["test_mean"]:.3f}', f'{m["test_std"]:.3f}'])
table(rows, [1.15, 1, 1.1, 1, 1.2, 1.2])
p('The standard deviation uses ddof=0. Test scores are sums of rewards, without discounting. The training objective uses discounted rewards. Each test uses a frozen stochastic policy for 500 full episodes.')
p('Main findings', 'heading')
p('The shuttle methods agree on all 201 states. The infinite-horizon policy sends a shuttle when at least 13 people are waiting. The EMA baseline improves CartPole in this run. It improves Pong earlier in training, but the final Pong test score is slightly lower. Neither final Pong policy wins a test game.')
p('How to read this report', 'heading')
p('Pages 2-4 cover the shuttle model, plots, and checks. Pages 5-10 explain policy gradient and show the required game plots. Pages 11-12 separate correctness checks from tuning experiments. Pages 13-14 give run commands and the three-month validation plan.')

page('2. Shuttle model and algorithms')
p('State s is the number of people waiting just before a decision: s = 0, ..., 200. Action a = 0 means wait; a = 1 means send one shuttle. The shuttle capacity is 15. Arrivals are independent and uniformly distributed over 1, 2, 3, 4, and 5.')
code('remaining(s,a) = max(s - 15*a, 0)\ncost(s,a) = 100*a + 2*remaining(s,a)\nnext_state = min(200, remaining(s,a) + arrivals)\ngamma = 0.95')
p('Order and assumptions', 'heading')
p('The agent observes the queue, chooses an action, and pays the dispatch and waiting costs. The waiting cost counts only people left after dispatch. New people arrive before the next decision. Arrivals above 200 are excluded from the queue; no extra penalty is charged. This cap is a modeling assumption, not an extra service mechanism. Sending an empty shuttle is allowed and costs 100. The terminal cost at time T is zero.')
p('Bellman update', 'heading')
code('Q(s,a) = cost(s,a)\n         + gamma * mean(V[next_state] for arrivals in 1..5)\nV(s) = min(Q(s,0), Q(s,1))')
p('The expectation averages all five next-state values. It does not replace random arrivals with their mean of 3: the value function is not generally linear.')
table([
    ['Method', 'What the code does', 'Stopping rule'],
    ['Finite horizon, T=500', 'Set V[500,s]=0. For t=499 down to 0, compare both actions at every state. Reuse the next row instead of expanding a full tree.', 'Finish all 500 rows. Plot V[0,s].'],
    ['Value iteration', 'Start all values at zero. Each sweep reads the old value array and minimizes over both actions.', 'Maximum absolute change over all states < 1e-6.'],
    ['Policy iteration', 'Start with wait everywhere. Evaluate the fixed policy, then improve every state using that same value array.', 'Evaluation tolerance 1e-8; stop when no action changes.'],
], [1.1, 2.5, 1.4])
p('On exact ties, finite-horizon and value iteration choose wait. Policy iteration keeps the old action. The finite-horizon result is exact for the stated discrete model up to floating-point error. Infinite-horizon results are numerical approximations controlled by their tolerances.')
p('Files: main.py; value_iteration.py; policy_iteration/main.py.', 'small')

page('3. Shuttle value functions')
figure('finite_horizon_value.png', 'Figure 1. Minimum expected discounted cost at time 0, T=500. The value rises as more passengers need service.', 284)
figure('value_iteration_value.png', 'Figure 2. Infinite-horizon value iteration, gamma=0.95. The shape is close to Figure 1 because distant costs have very small discount weights.', 284)
p('Value iteration stopped after 340 sweeps. The maximum difference between finite T=500 and value iteration was 1.81e-5 over all 201 states. Costs are not simulation averages; the update explicitly averages the five possible arrivals.', 'small')

page('4. Shuttle policy and validation')
figure('policy_iteration/optimal_policy.png', 'Figure 3. Infinite-horizon optimal action: wait below 13 passengers and dispatch at 13 or more. This result depends on the stated costs, capacity, cap, and discount.', 265)
p('Tests with known or independently checked answers', 'heading')
table([
    ['Test', 'Expected answer and reason', 'Observed result'],
    ['Set gamma=0', 'Only this step matters. Sending costs at least 70 more than waiting, so every state should wait and V(s)=2s.', '201/201 actions correct; maximum value error 0.'],
    ['Empty / partial / full shuttle', 'At s=0, sending costs100. At s=15, sending leaves0 and costs100. At s=16, it leaves1 and costs102.', 'All 7 hand-calculated cost checks passed.'],
    ['Queue cap', 'From remaining198, next states are199,200,200,200,200. Probability at200 must be4/5.', 'All 3 transition-expectation checks matched.'],
    ['Always wait at the cap', 'V(200)=400/(1-0.95)=8000. Neighbor states198/199 also have analytic values.', 'Maximum error 1.89e-7; allowed error2e-7.'],
    ['Compare three algorithms', 'Long finite horizon and two infinite-horizon methods should be close under this discount.', '201/201 actions agree for every pair; maximum value difference1.81e-5.'],
], [1.05, 2.3, 1.6])
p('Policy iteration stopped after 6 policy-evaluation/improvement rounds. Matching algorithms alone could hide a shared bug, so the zero-discount and analytic boundary tests provide separate checks.', 'small')
p('Evidence: policy_iteration/validation_gamma0.json, validation_boundaries.json, validation_comparison.json. Run: python policy_iteration/validate.py --experiment all.', 'small')

page('5. Policy gradient and baseline')
p('I use one hidden ReLU layer and two output logits. A categorical distribution turns logits into action probabilities. During both training and testing, actions are sampled from these probabilities. I use Adam and update the policy once after each episode.')
table([
    ['Setting', 'CartPole-v1', 'ALE/Pong-v5'],
    ['Input / actions', '4 state values; left or right', '6400 frame-difference pixels; actions2/3 only'],
    ['Network', '4 -> 32 -> 2', '6400 -> 128 -> 2'],
    ['Discount / learning rate', '0.95 / 0.01', '0.99 / 0.001'],
    ['Training episodes', '2000 per baseline setting', '5000 per baseline setting'],
    ['Initialization', 'PyTorch default', 'He-normal hidden weights; zero biases; orthogonal output weights, gain0.01'],
    ['Device', 'Local CPU', 'Deepdish4, one RTX A6000, visible GPU0'],
], [1.15, 1.55, 2.15])
p('Pong images are cropped to rows35:195, downsampled to80x80, stripped of two background colors, and made binary. I subtract the previous processed frame to show motion. At reset the previous frame is zero. I keep v5 defaults, including frameskip4 and sticky-action probability0.25. There is no reward shaping or reset of returns after individual Pong points.')
code('G[T] = 0\nG[t] = r[t] + gamma * G[t+1]       # work backward\nL = -sum_t gamma^t * (G[t] - b) * log pi(a[t] | s[t])')
p('The outer gamma^t is retained in the final experiments. For a long episode, gradients are accumulated in chunks of256 states, then Adam takes one step. No clipping, return normalization, critic network, or entropy bonus is used.')
p('Baseline: one historical moving average', 'heading')
code('b = 0                             # at the start of training\n# For each episode:\nupdate_policy_using(G - b)         # use the previous b\nreference = sum(gamma^t * G[t]) / sum(gamma^t)\nb = 0.9*b + 0.1*reference          # used in the next episode')
p('Without a baseline, b stays zero. With EMA, b is a scalar and is not learned by backpropagation. It is updated after the policy step, so it does not use the current action when serving as the current baseline. It can reduce gradient variance without guaranteeing a higher final score.')
p('Episodes end on either termination or time-limit truncation; no value is bootstrapped beyond that endpoint. Test runs freeze both the policy and the baseline. The reward at an individual step is supplied by the environment, not chosen by hand.')

for index, (name, title, train_note, result_note) in enumerate([
    ('cartpole_none', '6. CartPole without baseline', '2000 training episodes; gamma=0.95; Adam lr=0.01.', 'The final policy has a mean score of123.094. Training improves, but the policy does not reliably keep the pole up for500 steps.'),
    ('cartpole_ema', '7. CartPole with EMA baseline', 'Same initial seed, network, learning rate, and 2000-episode budget.', 'The final mean is369.722, compared with123.094 without baseline. The gain is246.628 points in this pair of runs; this is not a multi-seed result.'),
    ('pong_final_none', '8. Pong without baseline', '5000 training episodes; gamma=0.99; Adam lr=0.001; 128 hidden units.', 'The model improves late in training, but all500 final test games are losses. Mean=-18.374; standard deviation=2.743.'),
    ('pong_final_ema', '9. Pong with EMA baseline', 'Same starting weights and5000-episode budget; only the EMA baseline is added.', 'The baseline helps earlier in training, but all500 final test games are losses. Mean=-18.786; standard deviation=2.709.'),
]):
    page(title)
    p(train_note)
    figure(f'policy_gradient/results/{name}/training_rewards.png', f'Figure {4+2*index}. Training: each episode score and the trailing100-episode moving average. The first average is at episode100.', 263)
    figure(f'policy_gradient/results/{name}/evaluation_rewards.png', f'Figure {5+2*index}. Final test:500 frozen-policy episodes. The histogram uses raw episode reward; standard deviation uses ddof=0.', 263)
    p(result_note)

page('10. Comparing the two baselines')
figure('policy_gradient/cartpole_baseline_comparison.png', 'Figure 12. CartPole: EMA reaches a higher training level in this matched pair.', 248)
figure('policy_gradient/pong_baseline_comparison.png', 'Figure 13. Pong: EMA improves earlier. The no-baseline model catches up toward the end.', 248)
p('The Pong final mean difference (EMA minus no baseline) is -0.412. A paired bootstrap over the same500 test seeds gives a95% interval of[-0.706,-0.130]. Thus no baseline is slightly better for these two fixed models on this test. This interval does not include variation from retraining with other seeds.')
p('Pong mean discounted test return is-0.204 with EMA and-0.264 without it. EMA is better on this discounted metric but worse on whole-game score. The two measurements answer different questions. Final tests use seeds1000000-1000499; the50-game tuning checks use200000-200049.')

page('11. Correctness checks and edge cases')
p('A correctness test asks whether the algorithm gives the answer predicted by its equations. A tuning test asks whether a setting gives better results. Good tuning scores alone do not prove that gradients or transitions are correct.')
table([
    ['Check', 'Prediction / purpose', 'Saved evidence'],
    ['Exact two-step gradient', 'Enumerate4 action sequences. Compare the expected REINFORCE gradient with the direct gradient of J=p(0)+0.95*2*p(1). Repeat for b=0 and b=0.8.', 'Maximum errors8.94e-8 and1.49e-8, both below1e-6. Supports the sign, return formula, outer discount, and action-independent baseline.'],
    ['Chunked vs full gradient', 'Splitting an episode for memory should preserve its summed gradient.', 'Relative error8.46e-8 on an887-step Pong trajectory. A small gradient-descent step also lowered the fixed-trajectory loss.'],
    ['Paired gradient variance', 'Use20 separate CartPole pilot episodes to set b. Freeze the policy and compare both gradients on the same40 new trajectories.', 'Sum of component sample variances:891.86 without b;151.81 with b. Ratio0.1702. This checks variance directly, not score variance.'],
    ['Fixed-seed Pong replay', 'Keep policy, environment seed, and action seed fixed. Actions and rewards should repeat.', 'Both full946-step trajectories matched; score=-21. This is a reproducibility check, not proof of learning.'],
    ['Resume and test isolation', 'A split run should match uninterrupted training. Old scores must not be reused after a model changes.', 'CartPole EMA histories, weights, baseline and tests matched exactly; deliberately stale scores were rejected.'],
], [1.1, 2.2, 2.25])
p('Task-specific boundaries', 'heading')
p('Pong inputs contain only the intended foreground pixels; preprocessing leaves the original image unchanged. The code maps network action0 to environment action2, and1 to3. At every reset the previous frame is cleared. Both episode-ending flags are handled. Log probabilities are computed from logits for numerical stability, and non-finite losses are rejected. These implementation safeguards support, but do not replace, the algorithm checks above.')
p('First10 steps of the saved fixed-seed Pong check', 'heading')
trace = checks['pong']['first_10_steps']
table([['Step'] + list(range(1, 11)),
       ['Action'] + [r['environment_action'] for r in trace],
       ['Reward'] + [int(r['reward']) for r in trace],
       ['P(a=1)'] + [f'{r["probabilities"][1]:.3f}' for r in trace]])
p('Early rewards are zero because no point has yet been scored. The trace shows valid stochastic actions, not a manually assigned return. Evidence: policy_gradient/validation_results.json, resume_validation.json, tuning_results/audit.json and validation_checks.json.', 'small')

page('12. Tuning and weak Pong performance')
p('All tuning observations below are separate from the final500-game tests. I kept experiment names, seeds, model states, and decisions. I changed one main factor at a time where possible. These runs are useful evidence, but their repeated use of a small validation set limits model selection.')
table([
    ['Experiment', 'Budget and observation', 'Interpretation'],
    ['Learning rate', 'At100 episodes, lr0.01 had mean-20.90 and72.2% saturated actions in the early-step probe. lr0.0001 had mean-20.55. These used20 validation games.', 'The suggested0.01 was not a clearly good Pong setting. The final runs use0.001, a documented choice, not a proven optimum.'],
    ['Network width', 'At300 episodes and50 validation games:32=-20.42;64=-20.40;128=-20.32;256=-20.84, all with default initialization.', 'More hidden units did not automatically improve this short run. This does not settle the long-run comparison.'],
    ['Initialization', '128 with He hidden/small output initialization scored-20.70 at300 episodes.', 'No short-run advantage. I retained it for a controlled longer-budget test.'],
    ['Longer training', 'The same no-baseline model improved from-20.70 at300 to-18.60 at5000 on50 tuning games.', '300 episodes were not enough to show its later improvement.'],
    ['EMA baseline', 'At5000,50-game tuning means were-18.60 without and-18.42 with EMA; their score-difference interval crossed0.', 'No clear tuning advantage. The independent final500-game test favors the no-baseline model slightly.'],
    ['Remove outer discount', 'A separate historical branch omitted gamma^t from the loss. At600 episodes it scored-19.80 on50 tuning games.', 'This changes the gradient estimator. The final runs restore the original formula; the branch is not relabeled as the final method.'],
], [1.1, 2.4, 1.8])
p('Why can the score remain poor?', 'heading')
p('Pong rewards are sparse and games are long. With gamma=0.99, a reward500 steps later is weighted by about0.00657; at1000 steps the weight is about0.0000432. The objective puts much more weight on early events. Better discounted return may therefore fail to improve the full-game score.')
p('Exploration also decreased. In the50-game tuning probes, the share of first256-step action probabilities above99% rose to40.5% without a baseline and73.4% with EMA. This is a possible reason for limited improvement, not a proven causal explanation. The one-layer pixel policy and per-episode updates may also limit learning. Further experiments would be needed to separate these causes.')
p('I do not claim that the final policies solve Pong. I also do not claim that EMA always reduces test-score variation or always improves learning. The direct variance test and the final performance comparison measure different things.')

page('13. How to run the code')
p('All paths below are relative to the hw1 folder in the repository. Keep the existing results as evidence and use new output folders for new runs. Install PyTorch for your machine; then install the small dependency files. The report itself needs reportlab.')
code('python -m pip install numpy matplotlib\npython -m pip install -r policy_gradient/requirements.txt\npython -m pip install reportlab\n\n# Shuttle answers and validation\npython main.py\npython value_iteration.py\npython policy_iteration/main.py\npython policy_iteration/validate.py --experiment all')
p('Reproduce the game methods and training budgets', 'heading')
code('cd policy_gradient\n# Run these commands once for b=none and once for b=ema.\n# Replace BASELINE with none or ema.\npython train.py --game cartpole --baseline BASELINE \\\n  --episodes 2000 --eval-episodes 500 --lr 0.01 \\\n  --device cpu --output results/new_cartpole_BASELINE\n\nCUDA_VISIBLE_DEVICES=0 python train.py --game pong \\\n  --baseline BASELINE --episodes 5000 --eval-episodes 500 \\\n  --hidden 128 --initialization he_small_output --lr 0.001 \\\n  --device cuda --output results/new_pong_BASELINE')
p('The saved Pong runs used tune_pong.py for training and evaluate_pong_final.py for the final tests. The no-baseline run resumed from300 episodes with its Adam state intact; the EMA run started fresh. Both began with identical seed0 weights. The commands above reproduce the method and budget, not a promise of bitwise equality across devices or library versions.')
p('Run and inspect the validation experiments', 'heading')
code('# From policy_gradient/\npython validate.py\npython validate_tuning.py\npython validate_loss_modes.py\npython test_resume.py\npython summarize.py --require-complete\n\n# From hw1/: rebuild this PDF from saved JSON and figures\npython build_report.py')
p('validate.py checks the exact gradient, Pong replay/preprocessing, and paired CartPole gradient variance. validate_tuning.py checks the exact gradient and batch-gradient averaging. validate_loss_modes.py checks both historical loss forms. test_resume.py is an additional engineering check. The original GPU audit can be rerun with CUDA_VISIBLE_DEVICES=0 python tune_pong.py --audit --output smoke/recheck_audit; it uses20 tuning-validation games, not final test seeds.', 'small')
p('Environment record: CartPole used PyTorch2.14.1 on CPU, Gymnasium1.4.0, NumPy2.5.4. Pong used PyTorch2.14.1+cu126, Gymnasium1.4.0 and ALE0.12.1 on one RTX A6000. Deepdish used the myenv conda environment. No multiple-GPU training was used.', 'small')

page('14. If I had three more months')
p('The goal would be to learn which conclusions are repeatable, why Pong remains weak, and where the code could fail. I would keep correctness tests separate from model selection and keep a new final test set untouched until the settings are fixed.')
table([
    ['Period', 'Tests and experiments', 'What I would measure'],
    ['Month1: reliability', 'Repeat each game/baseline setting with at least5 training seeds. Add short problems with exact values and exact gradients. For the shuttle, test deterministic arrivals, zero arrival rate, zero dispatch cost, changed capacity, and alternative queue-cap rules.', 'Mean and uncertainty across training seeds; exact-solution errors; Bellman residuals and policy agreement. Separate repeated-run variation from test-game variation.'],
    ['Month2: controlled comparisons', 'Use matched budgets to change one factor at a time: learning rate, batch size, hidden width, initialization, or baseline. Compare a constant baseline, historical EMA, and a simple state-dependent baseline. Measure gradient variance at several frozen checkpoints on shared trajectories.', 'Learning speed, final score, discounted return, gradient variance, action entropy, time and memory. Do not assume that a smaller gradient variance guarantees better scores.'],
    ['Month3: robustness and final evaluation', 'Test unusual starting states and near-terminal states. For diagnostics, compare deterministic settings with the normal stochastic game. Study very long games and the effect of discounting. Then freeze the selected method and use new final test seeds.', 'Performance failures, robustness to changes, confidence intervals, and whether improvements survive independent testing. Keep the required v5 setup for the main result.'],
], [1.1, 2.8, 2.1])
p('Limits of the current evidence', 'heading')
p('The current results use one training seed. The500-game tests describe these trained models, not every run of the algorithm. The tuning sets were reused, so small tuning differences can be misleading. A tiny analytic example strongly checks a gradient formula but cannot prove every part of a large program correct. Replay checks test reproducibility, not whether a policy is good.')
p('The main code is basic: arrays and loops for the shuttle, and a two-layer network with one optimizer step per episode for policy gradient. The extra scripts save results, run checks, and draw figures. No new training or parameter changes were made while writing this report.')
p('Sources and saved evidence', 'heading')
p('Task requirements: Assignment1, sections1-4, and MLDS Lab Week1 setup instructions. No external performance target was added. The report uses the recorded data listed below; no test outcomes were invented.')
code('policy_iteration/validation_*.json\npolicy_gradient/validation_results.json\npolicy_gradient/resume_validation.json\npolicy_gradient/tuning_results/audit.json\npolicy_gradient/tuning_results/validation_checks.json\npolicy_gradient/tuning_results/stage_decisions.json\npolicy_gradient/results/cartpole_{none,ema}/metrics.json\npolicy_gradient/results/pong_final_{none,ema}/metrics.json\npolicy_gradient/final_test_comparison.json')
p('Final Pong tests are tied to the selected model fingerprints and checkpoint hashes in tuning_results/final_test_500.json and final_test_verification.json. Raw scores, figures, code, and this report are included in the repository; large model files and local environments are kept outside Git.', 'small')


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#cbd3da'))
    canvas.line(44, 34, A4[0]-44, 34)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#53616d'))
    canvas.drawString(44, 22, 'MLDS Assignment 1 | jingliangguo2027')
    canvas.drawRightString(A4[0]-44, 22, str(doc.page))


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=44, rightMargin=44,
                        topMargin=40, bottomMargin=44, title='Assignment 1: Reinforcement Learning',
                        author='jingliangguo2027')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
