"""Build the report from saved results. This script does not train any model."""
from pathlib import Path
import json
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
    return str(text)


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
    import statistics
    assert len(m['test_scores']) == 500
    assert abs(statistics.mean(m['test_scores']) - m['test_mean']) < 1e-9
    assert abs(statistics.pstdev(m['test_scores']) - m['test_std']) < 1e-9
evidence = read('policy_gradient/evidence.json')
checks = evidence['validation']
shuttle = read('policy_iteration/validation_comparison.json')


variance = evidence['pong_variance']

page('assignment1')
p('Jingliang Guo', 'heading')
p('netid: nxt1510')
p('1. Shuttle: model and methods', 'heading')
p('State s is the number waiting, from 0 to 200. Action a is 0 (wait) or 1 (dispatch). Capacity is 15. Independent arrivals are uniform over {1, 2, 3, 4, 5}.')
code('remaining(s,a) = max(s - 15*a, 0)\ncost(s,a) = 100*a + 2*remaining(s,a)\nnext_state = min(200, remaining(s,a) + arrivals)\ngamma = 0.95')
p('I charge waiting cost after dispatch, then add arrivals. Arrivals above 200 leave the system without a penalty. An empty shuttle may be sent for cost 100. The finite-horizon terminal cost is zero.')
code('Q_t(s,a) = cost(s,a) + gamma * mean(V_(t+1)[next_state])\nV_t(s) = min(Q_t(s,0), Q_t(s,1))')
p('The mean uses all five arrival outcomes, each with probability 1/5.')
table([
    ['Part', 'Method'],
    ['(a) Enumeration, T=500', 'I enumerate both actions and all five arrival outcomes at every state and time. Backward dynamic programming reuses the computed future values: set V_500=0, then solve t=499,...,0. This avoids expanding the full trajectory tree.'],
    ['(b) Value iteration', 'Start V=0. Apply the Bellman update using the previous sweep until the largest change is below 1e-6.'],
    ['(c) Policy iteration', 'Start with wait everywhere. Evaluate the policy to tolerance 1e-8, then choose the cheaper action at each state. Stop when no action changes.'],
], [1, 3])
p('Part (a) interprets enumeration as checking every state-action-outcome combination, rather than listing complete policies. On ties, (a) and (b) wait; (c) keeps its current action.', 'small')
p('Code: main.py, value_iteration.py, policy_iteration/main.py.', 'small')

page('2. Shuttle: value functions')
figure('finite_horizon_value.png', 'Figure 1. Part (a): minimum discounted cost at time 0, T=500.', 290)
figure('value_iteration_value.png', 'Figure 2. Part (b): minimum discounted cost for the infinite horizon.', 290)
p('More waiting passengers increase cost. The curves nearly overlap because distant costs receive little weight at gamma=0.95. Value iteration took 340 sweeps; its largest difference from T=500 was 1.81e-5.')

page('3. Shuttle: policy and checks')
figure('policy_iteration/optimal_policy.png', 'Figure 3. Part (c): wait below 13 passengers; dispatch at 13 or more.', 265)
p('Policy iteration stopped after 6 rounds. All three methods agree on the action at every one of the 201 states.')
table([
    ['Test', 'Expected result', 'Observed result'],
    ['gamma=0', 'Ignore future costs. Dispatch costs at least 70 extra, so always wait; V(s)=2s.', '201/201 actions correct; value error 0.'],
    ['Empty / partial / full', 'Dispatch at s=0 or 15 costs 100; at s=16 it costs 102.', 'All 7 hand-calculated cost checks pass.'],
    ['Queue cap', 'From 198 remaining, next states are 199,200,200,200,200; P(200)=4/5.', 'All 3 expectation checks pass.'],
    ['Always wait', 'At the cap, V(200)=400/(1-0.95)=8000. Values at 198 and 199 also have analytic solutions.', 'Largest error 1.89e-7; tolerance 2e-7.'],
    ['Compare methods', 'T=500 should be close to the infinite-horizon solutions.', '201/201 actions agree; largest value gap 1.81e-5.'],
], [1, 2.1, 1.5])
p('The analytic checks provide independent answers; agreement between implementations alone could miss a shared error.', 'small')
p('Run: python policy_iteration/validate.py --experiment all. Results: policy_iteration/validation_*.json.', 'small')

page('4. Policy gradient: method')
p('I use REINFORCE with one ReLU hidden layer. The network outputs two action logits. Actions are sampled during training and testing. Adam updates the policy once per episode.')
table([
    ['Setting', 'CartPole-v1', 'ALE/Pong-v5'],
    ['Input / actions', '4 state values; actions 0/1', '6400 frame-difference pixels; actions 2/3'],
    ['Network', '4 -> 32 -> 2', '6400 -> 128 -> 2'],
    ['Discount / Adam rate', '0.95 / 0.01', '0.99 / 0.001'],
    ['Training episodes', '2000 per setting', '5000 per setting'],
    ['Initialization', 'PyTorch default', 'He hidden weights; zero biases; orthogonal output, gain 0.01'],
    ['Device', 'CPU', 'One RTX A6000 on Deepdish'],
], [1.3, 1.45, 2.25])
p('For Pong, I use the supplied crop, downsampling and binary preprocessing, then subtract consecutive frames to show motion. The previous frame is zero at reset. I retain v5 defaults: frameskip 4 and sticky-action probability 0.25.')
code('G[T] = 0\nG[t] = r[t] + gamma * G[t+1]\nloss = -sum_t gamma^t * (G[t] - b) * log pi(a[t] | s[t])')
p('Returns continue across Pong points. Gradients are summed in 256-state chunks before one Adam step. Episodes end on termination or truncation, with no bootstrap value.')
p('Part 2: historical EMA baseline', 'heading')
code('b = 0\n# Each episode: use the old b in the loss above.\nupdate_policy()\nreference = sum_t(gamma^t * G[t]) / sum_t(gamma^t)\nb = 0.9*b + 0.1*reference')
p('Part 1 uses b=0 throughout. Part 2 uses a scalar baseline from previous episodes, independent of the current action. Both settings use the same starting weights and training budget for each game.')
p('Evaluation', 'heading')
p('Each trained policy is frozen and tested for 500 episodes. Scores are undiscounted reward sums; standard deviations use ddof=0. Training plots show episode rewards and a trailing 100-episode average, starting at episode 100.')
rows=[['Game', 'Baseline', 'Test mean', 'Test std. dev.']]
for name,m in metrics.items():
    rows.append(['CartPole' if name.startswith('cartpole') else 'Pong', m['config']['baseline'], f'{m["test_mean"]:.3f}', f'{m["test_std"]:.3f}'])
table(rows)

for index, (name, title, note) in enumerate([
    ('cartpole_none', '5. CartPole: no baseline', 'The mean test score is 123.094. Learning is unstable, and the policy rarely reaches the 500-step limit.'),
    ('cartpole_ema', '6. CartPole: EMA baseline', 'The mean test score is 369.722, an increase of 246.628 over no baseline in this pair of runs.'),
    ('pong_final_none', '7. Pong: no baseline', 'The mean test score is -18.374. Scores improve late in training, but all 500 test games are losses.'),
    ('pong_final_ema', '8. Pong: EMA baseline', 'The mean test score is -18.786. EMA improves earlier in training, but all 500 test games are losses.'),
]):
    page(title)
    figure(f'policy_gradient/results/{name}/training_rewards.png', f'Figure {4+2*index}. Training rewards and the 100-episode moving average.', 280)
    figure(f'policy_gradient/results/{name}/evaluation_rewards.png', f'Figure {5+2*index}. Reward distribution over 500 frozen-policy test episodes.', 280)
    p(note)

page('9. Policy gradient: validation')
table([
    ['Check', 'Method and expected result', 'Result'],
    ['Exact gradient', 'Enumerate all 4 two-step trajectories. Compare expected REINFORCE gradients with direct differentiation of J=p(0)+0.95*2*p(1), for b=0 and b=0.8.', 'Maximum errors 8.94e-8 and 1.49e-8, below 1e-6. Checks the gradient sign, discounts and baseline invariance.'],
    ['Chunked gradient', 'Compare chunked and full gradients on the same 887-step Pong trajectory.', 'Relative error 8.46e-8. A small gradient step also lowers the fixed-trajectory loss.'],
    ['Pong replay', 'Freeze the policy and repeat the environment and action seeds.', 'Both 946-step action/reward traces match; score -21.'],
], [1, 2.2, 2])
p('Does the baseline reduce gradient variance?', 'heading')
p('For each game, freeze an untrained policy and estimate b from 20 separate pilot episodes. On each new trajectory, compute gradients both with and without b. Measure the sum of sample variances across all gradient components.')
v=checks['gradient_variance']['sum_of_component_gradient_variances']
vp=variance['statistics']
table([
    ['Game', 'Paired episodes', 'No baseline', 'EMA', 'EMA / none'],
    ['CartPole', checks['gradient_variance']['paired_episodes'], f'{v["none"]:.2f}', f'{v["historical_ema"]:.2f}', f'{checks["gradient_variance"]["variance_ratio_ema_to_none"]:.4f}'],
    ['Pong', variance['paired_episodes'], f'{vp["none"]["sum_component_variances"]:.2f}', f'{vp["historical_ema"]["sum_component_variances"]:.2f}', f'{variance["variance_ratio"]:.4f}'],
], [1, 1.2, 1.2, 1, 1.1])
p('Variance falls by about 83% for CartPole and 82% for Pong in these checks. The Pong check uses a seed-0, default-initialized 128-unit policy, not either final trained model. This supports variance reduction at those fixed policies; it does not guarantee higher scores.')
p('Edge cases', 'heading')
p('Pong preprocessing preserves the original image and produces binary frames. Network actions 0/1 map to environment actions 2/3. Frame history resets each episode. Both termination flags are handled; log probabilities use logits, and non-finite losses are rejected.')
p('In the saved Pong replay, the first 10 actions are 3,3,3,3,2,2,3,3,2,2. All 10 rewards are zero because no point has been scored.', 'small')
p('Evidence: policy_gradient/evidence.json. Validation commands are on page 11.', 'small')

page('10. Experiments and discussion')
table([
    ['Pong experiment', 'Observation', 'Conclusion'],
    ['Learning rate', 'After 100 episodes: lr=0.01 scored -20.90; lr=0.0001 scored -20.55 on 20 validation games. The former had 72.2% saturated probe actions.', 'The suggested rate was unstable. I used 0.001 for the final runs.'],
    ['Hidden width', 'At 300 episodes, 50-game means for 32/64/128/256 units were -20.42/-20.40/-20.32/-20.84.', 'More units did not consistently help.'],
    ['Initialization', '128 units with He/small-output initialization scored -20.70 at 300 episodes.', 'No early advantage; retained for the longer run.'],
    ['Training length', 'That no-baseline model improved from -20.70 at 300 episodes to -18.60 at 5000 on 50 validation games.', 'Longer training helped.'],
    ['EMA baseline', 'At 5000 episodes, validation means were -18.60 without and -18.42 with EMA; the difference interval crossed zero.', 'No clear validation advantage.'],
    ['Historical loss variant', 'Omitting the outer gamma^t scored -19.80 after 600 episodes on 50 validation games.', 'This changes the estimator. Final runs retain gamma^t.'],
], [1.05, 2.7, 1.55])
p('Final comparison and limits', 'heading')
p('CartPole improves with EMA in this run. For Pong, EMA minus no baseline is -0.412 points; a paired bootstrap over 500 test seeds gives a 95% interval of [-0.706, -0.130]. Mean discounted return favors EMA (-0.204 vs -0.264), but whole-game score does not. These results use one training seed, so they do not establish which method is better across retraining.')
p('Pong rewards are sparse. At gamma=0.99, rewards 500 steps away receive weight 0.00657. Exploration also drops: 40.5% of no-baseline and 73.4% of EMA probe actions have probability above 99%. These may explain weak scores, but are not causal tests. The reused tuning set also limits model selection.')
p('With three more months', 'heading')
table([
    ['Period', 'Experiments and measurements'],
    ['Month 1', 'Repeat all settings with at least 5 training seeds and report uncertainty. Test shuttle deterministic/zero arrivals, free dispatch, changed capacity and cap rules against exact solutions.'],
    ['Month 2', 'Compare learning rates, batch sizes, widths and baseline types at matched budgets. Measure scores, learning speed and paired gradient variance at several checkpoints.'],
    ['Month 3', 'Test unusual and near-terminal states, deterministic diagnostics and longer games. Freeze selected settings and evaluate on fresh test seeds; report failures and confidence intervals.'],
], [1, 4])

page('11. Code and reproduction')
p('GitHub username: jingliangguo2027<br/>Repository: mlds-assignment-1<br/><link href="https://github.com/jingliangguo2027/mlds-assignment-1" color="#173957">https://github.com/jingliangguo2027/mlds-assignment-1</link>', 'small')
p('Start in hw1. Install PyTorch for your CPU or CUDA setup, then run:')
code('python -m pip install -r requirements.txt\npython -m pip install reportlab\npython main.py\npython value_iteration.py\npython policy_iteration/main.py\npython policy_iteration/validate.py --experiment all')
p('Train and test', 'heading')
p('Run each command with BASELINE replaced by none, then ema. Use new output folders.')
code('cd policy_gradient\npython train.py --game cartpole --baseline BASELINE \\\n  --episodes 2000 --eval-episodes 500 --lr 0.01 \\\n  --device cpu --output results/new_cartpole_BASELINE\n\nCUDA_VISIBLE_DEVICES=0 python train.py --game pong \\\n  --baseline BASELINE --episodes 5000 --eval-episodes 500 \\\n  --hidden 128 --initialization he_small_output --lr 0.001 \\\n  --device cuda --output results/new_pong_BASELINE')
p('The runner keeps the original method and seeds: Pong action seeds are 10000+episode and environment seeds are episode. Exact results can vary across devices or library versions. --resume continues a new-format checkpoint; old checkpoints remain in the external backup.', 'small')
p('Validation commands (from policy_gradient)', 'heading')
code('python validate.py\nCUDA_VISIBLE_DEVICES=0 python validate.py --extra pong-variance --device cuda\nCUDA_VISIBLE_DEVICES=0 python validate.py --extra chunk --device cuda')
p('The default checks exact gradients, Pong preprocessing/replay and CartPole gradient variance. The extra modes check Pong variance and chunked gradients against a full-trajectory gradient. They print fresh results without replacing the saved evidence.', 'small')
p('Saved data and environment', 'heading')
p('Raw training and test scores, settings and plots are in policy_gradient/results/. Evidence and selected tuning summaries are consolidated in policy_gradient/evidence.json. Model files are not in Git. Final test seeds: 1000000-1000499; tuning seeds: 200000-200049.', 'small')
p('Recorded environment: PyTorch 2.14.1 (Pong: +cu126), Gymnasium 1.4.0, NumPy 2.5.4 and ALE 0.12.1. Pong used one GPU. From hw1, run python build_report.py to validate the four recorded score sets and rebuild this PDF.', 'small')
p('Historical tuning scripts and original checkpoint formats remain available in <link href="https://github.com/jingliangguo2027/mlds-assignment-1/tree/e8ec206580c0c26dac98153dbd1bf1438fd36f4b/hw1/policy_gradient" color="#173957">Git history (e8ec206)</link>. The current code keeps the assignment methods and merges the necessary validation checks.', 'small')



def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#53616d'))
    canvas.drawRightString(A4[0]-44, 22, str(doc.page))


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=44, rightMargin=44,
                        topMargin=40, bottomMargin=44, title='assignment1',
                        author='Jingliang Guo')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
