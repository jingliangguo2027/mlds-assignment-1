"""Build the report and figures from recorded experiment data."""
import json
from pathlib import Path
import statistics
import sys
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, Preformatted, KeepTogether)

ROOT = Path(__file__).resolve().parent
PG = ROOT/'policy_gradient'
OUT = ROOT/'output/pdf/assignment1_report.pdf'
FIG = ROOT/'output/figures'
OUT.parent.mkdir(parents=True,exist_ok=True)
FIG.mkdir(parents=True,exist_ok=True)
WIDTH = A4[0]-88
BLUE = colors.HexColor('#173957')
styles = {
    'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=BLUE,spaceAfter=11),
    'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=11.5,leading=15,textColor=BLUE,spaceBefore=8,spaceAfter=5),
    'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=7),
    'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=6),
    'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.6,leading=11.6),
    'code':ParagraphStyle('code',fontName='Courier',fontSize=8,leading=10.8,spaceAfter=8),
}
story = []

def read(path):
    return json.loads((ROOT/path).read_text())

def p(text,style='body'):
    story.append(Paragraph(text,styles[style]))

def page(title):
    if story:
        story.append(PageBreak())
    p(title,'title')

def code(text):
    story.append(Preformatted(text,styles['code']))

def table(rows,widths=None):
    widths = widths or [1]*len(rows[0])
    cells = [[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in rows]
    t = Table(cells,colWidths=[WIDTH*w/sum(widths) for w in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),.8,BLUE),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#cbd3da')),
    ]))
    story.extend([t,Spacer(1,7)])

def figure(path,caption,height=260):
    img = Image(str(ROOT/path))
    scale = min(WIDTH/img.imageWidth,height/img.imageHeight)
    img.drawWidth,img.drawHeight = img.imageWidth*scale,img.imageHeight*scale
    story.append(KeepTogether([img,Spacer(1,3),Paragraph(caption,styles['small'])]))

metrics = {name:read(f'policy_gradient/results/{name}/metrics.json') for name in
           ('cartpole_none','cartpole_ema','pong_final_none','pong_final_ema')}
for m in metrics.values():
    assert len(m['test_scores'])==500
    assert abs(statistics.mean(m['test_scores'])-m['test_mean'])<1e-9
    assert abs(statistics.pstdev(m['test_scores'])-m['test_std'])<1e-9
new = read('policy_gradient/results/pong_standardized/metrics.json')
final_test = read('policy_gradient/results/pong_standardized/evaluation.json')
scores = new['training_scores']
assert len(scores)==new['training_episodes']==4000
assert abs(statistics.mean(scores[-100:])-new['last100_mean'])<1e-9
assert sum(s>0 for s in scores[-100:])==new['last100_wins']==89
assert new['independent_test_episodes']==500 and not new['endpoint_checkpoint_available']
assert new['independent_test_episodes_at_reported_endpoint']==0
assert new['source_checkpoint_episodes']==4160
assert final_test['training_episodes']==4160 and final_test['test_episodes']==500
assert len(final_test['test_scores'])==500
assert abs(statistics.mean(final_test['test_scores'])-final_test['test_mean'])<1e-9
assert abs(statistics.pstdev(final_test['test_scores'])-final_test['test_std'])<1e-9
assert sum(x>0 for x in final_test['test_scores'])==final_test['test_wins']
assert abs(final_test['test_win_rate']-final_test['test_wins']/500)<1e-12
assert final_test['source_checkpoint_sha256']==new['source_checkpoint_sha256']
evidence = read('policy_gradient/evidence.json')
checks = evidence['validation']
variance = evidence['pong_variance']


def original_panels(game,names):
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs = plt.subplots(2,2,figsize=(8.3,5.5),layout='constrained')
        for col,(name,label) in enumerate(zip(names,('No baseline','EMA baseline'))):
            m = metrics[name]
            history = np.asarray(m['training_scores'])
            x = np.arange(1,len(history)+1)
            ax = axs[0,col]
            ax.plot(x,history,lw=.5,alpha=.35,color='#9cb9cf')
            ax.plot(x[99:],np.convolve(history,np.ones(100)/100,'valid'),color='#c76022',lw=1.6)
            ax.set(title=f'{label}: training',xlabel='Training episode',ylabel='Episode reward')
            if game=='Pong':ax.set_ylim(-22,22)
            ax = axs[1,col]
            bins = np.arange(-21.5,22.5,1) if game=='Pong' else np.linspace(0,500,26)
            ax.hist(m['test_scores'],bins=bins,color='#417da1',edgecolor='white')
            ax.set(title=f'{label}: 500 test games',xlabel='Episode reward',ylabel='Count')
            ax.text(.04,.95,f'Mean {m["test_mean"]:.2f}\nSD {m["test_std"]:.2f}',
                    transform=ax.transAxes,va='top',fontsize=9)
        fig.savefig(FIG/f'{game.lower()}_comparison.png',dpi=180)
        plt.close(fig)


original_panels('CartPole',('cartpole_none','cartpole_ema'))
original_panels('Pong',('pong_final_none','pong_final_ema'))
sys.path.insert(0,str(PG))
from pong_standardized import plot
plot(PG/'results/pong_standardized')

page('assignment1')
p('Jingliang Guo','heading')
p('netid: nxt1510')
p('1. Shuttle: model, solution and checks','heading')
p('State s is the number waiting (0-200); action a is wait (0) or dispatch (1). Capacity is 15. Independent arrivals are uniform over {1,2,3,4,5}.')
code('remaining = max(s - 15*a, 0)\ncost = 100*a + 2*remaining\nnext_state = min(200, remaining + arrivals)\ngamma = 0.95')
p('Waiting cost is charged after dispatch and before arrivals. Excess arrivals leave without penalty. An empty dispatch costs 100. The finite-horizon terminal value is zero.')
code('Q_t(s,a) = cost(s,a) + gamma * mean(V_(t+1)[next_state])\nV_t(s) = min_a Q_t(s,a)')
p('<b>Enumeration:</b> check both actions and all five arrivals at every state and time, working backward from T=500. <b>Value iteration:</b> apply Bellman updates from V=0 until the largest change is below 1e-6. <b>Policy iteration:</b> alternate policy evaluation (tolerance 1e-8) and improvement until no action changes.')
p('Enumeration here means exhaustive state-action-outcome evaluation with reused future values. Ties wait in enumeration/value iteration; policy iteration keeps its current action.','small')
table([
    ['Validation / edge case','Expected result','Observed'],
    ['gamma=0','Always wait; V(s)=2s','201/201 actions correct; value error 0'],
    ['Empty / partial / full shuttle','Dispatch costs 100 at s=0 or 15; 102 at s=16','7 hand-calculated cases pass'],
    ['Queue cap','From 198 remaining, P(next=200)=4/5','3 expectation checks pass'],
    ['Always wait at the cap','V(200)=400/(1-0.95)=8000','Largest analytic error 1.89e-7'],
    ['Compare all three methods','Same policy; finite/infinite values close','201/201 actions agree; value gap 1.81e-5'],
],[1.2,2.1,1.9])
p('The analytic cases give independent answers; agreement between implementations alone could miss a shared error.','small')

page('2. Shuttle: results')
figure('finite_horizon_value.png','Figure 1. Enumeration: minimum expected discounted cost at time 0, T=500.',185)
figure('value_iteration_value.png','Figure 2. Infinite-horizon value iteration: optimal value versus queue size.',185)
figure('policy_iteration/optimal_policy.png','Figure 3. Policy iteration: wait below 13 passengers; dispatch at 13 or more.',185)
p('Larger queues cost more. The value curves nearly overlap at gamma=0.95. Value iteration takes 340 sweeps; policy iteration takes 6 rounds.','small')

page('3. Policy gradient: original method')
p('REINFORCE uses one ReLU hidden layer and two action logits. Actions are sampled in training and testing. The original four runs update Adam once per episode.')
table([
    ['Setting','CartPole-v1','ALE/Pong-v5'],
    ['Input / actions','4 state values; actions 0/1','6400 frame differences; actions 2/3'],
    ['Network','4 -> 32 -> 2','6400 -> 128 -> 2'],
    ['gamma / learning rate','0.95 / 0.01','0.99 / 0.001'],
    ['Training episodes per setting','2000','5000'],
    ['Initialization','PyTorch default','He hidden layer; output gain 0.01; zero biases'],
],[1.3,1.4,2.3])
p('Pong uses the supplied crop, downsampling and binary preprocessing, followed by consecutive-frame subtraction. Previous-frame memory resets each episode. The v5 defaults remain: frameskip 4 and sticky-action probability 0.25.')
p('I reduced the Pong learning rate to 0.001 for stability. In the 0.01 trial, 72.2% of probe states had an action probability above 99% after 100 training games, while the 20-game validation mean remained -20.90.','small')
code('G[T] = 0\nG[t] = r[t] + gamma*G[t+1]\nloss = -sum_t gamma^t * (G[t]-b) * log pi(a[t]|s[t])')
p('Returns continue across all Pong points. Both termination and truncation end an episode; there is no value bootstrap. Gradients accumulate in 256-state chunks before a single update.')
p('Baseline comparison','heading')
p('Part 1 uses b=0. Part 2 uses a scalar historical EMA, computed before the current episode. Each game uses matched starting weights, seeds and budgets.')
code('# After the policy update:\nreference = sum_t(gamma^t*G[t]) / sum_t(gamma^t)\nb = 0.9*b + 0.1*reference')
p('Each original final policy is frozen and tested for 500 games. Scores are undiscounted reward sums; standard deviations use ddof=0. These tests are separate from the follow-up training results on page 6.')
rows=[['Game','Baseline','Test mean','Test SD']]
for name,m in metrics.items():
    rows.append(['CartPole' if name.startswith('cartpole') else 'Pong',m['config']['baseline'],f'{m["test_mean"]:.3f}',f'{m["test_std"]:.3f}'])
table(rows)

page('4. CartPole: original results')
figure('output/figures/cartpole_comparison.png','Figure 4. Original CartPole runs. Top: episode rewards (blue) and trailing 100-episode means (orange). Bottom: 500-game frozen-policy test histograms.',410)
p('The EMA policy scores 369.722 versus 123.094 without a baseline, a gain of 246.628 in this pair of runs. Both learning curves fluctuate, so training improvement is not monotonic.')
p('This is one training seed per setting. The comparison supports an improvement for these two trained models, rather than a general ranking across retraining.')

page('5. Pong: original results')
figure('output/figures/pong_comparison.png','Figure 5. Original Pong runs. Top: episode rewards and trailing 100-episode means. Bottom: 500-game frozen-policy test histograms.',410)
table([
    ['Original setting','Training episodes','Test games','Test mean','Test win rate'],
    ['No baseline',5000,500,'-18.374','0%'],
    ['EMA baseline',5000,500,'-18.786','0%'],
],[1.4,1.2,1,1,1.1])
p('Both models improve late but still lose every test game. EMA reduces measured gradient variance (page 7), yet it does not improve the final whole-game score in these runs.')
p('Learning-rate, width, update-frequency and initialization trials did not produce a comparable improvement within their tested budgets. I therefore changed how returns enter the update in the separate follow-up experiment.')

page('6. Pong: standardized returns')
p('I standardize each episode\'s advantages to provide a relative performance reference and a consistent scale. I also remove the outer gamma^t weight, so late-game updates are not additionally suppressed. Full-game returns still use gamma=0.99; returns are <b>not reset at scored points</b>.')
code('A[t] = (G[t]-b - mean(G-b)) / (std(G-b) + 1e-8)\nloss = -sum_t A[t] * log pi(a[t]|s[t])')
p('Adam uses lr=0.001 and averages gradients over 10 games per update. The network remains 6400 -> 128 -> 2, with output gain 0.1 as in the preceding trial. The constant EMA cancels under centering, so this is not evidence of an EMA benefit.','small')
figure('policy_gradient/results/pong_standardized/training_rewards.png','Figure 6. Follow-up: episode rewards and the trailing 100-episode mean, episodes 1-4000.',195)
figure('policy_gradient/results/pong_standardized/training_win_rate.png','Figure 7. Training win rate over the most recent 100 games; a win has positive episode reward.',155)
table([['Episode','Last-100 mean','Last-100 win rate']]+[
    [m['episode'],f'{m["mean"]:+.2f}',f'{100*m["win_rate"]:.0f}%']
    for m in new['milestones'] if m['episode'] in (100,1000,2000,4000)
],[1,1,1.2])
p('<b>The combined changes improve training from -20.59 and 0% wins to +7.20 and 89% wins.</b> Standardization and outer-weight removal were changed together; this run does not isolate their effects.','small')
p('Training curves show the first 4000 games. The final checkpoint was saved at 4160; its independent 500-game test is on page 7. This follow-up changes the estimator; the original discounted-objective comparison is retained.','small')

page('7. Final test and validation')
p(f'The frozen 4160-game checkpoint scores <b>{final_test["test_mean"]:+.3f} (SD {final_test["test_std"]:.3f})</b> on 500 new games, winning <b>{final_test["test_wins"]}/500 ({final_test["test_win_rate"]:.1%})</b>. Actions are sampled from the policy; no parameters are updated.')
figure('policy_gradient/results/pong_standardized/evaluation_rewards.png',
       'Figure 8. Independent 500-game test of the final standardized-return Pong model.',145)
table([
    ['Original-method check','Expected behavior / procedure','Observed'],
    ['Exact gradient','Enumerate all four two-step trajectories; compare REINFORCE with direct differentiation, b=0 and b=0.8','Max errors 8.94e-8 and 1.49e-8'],
    ['Full vs chunked gradient','Same 887-step Pong trajectory; small descent step should lower its fixed-sample loss','Relative error 8.46e-8; loss decreases'],
    ['Seeded Pong replay','Fixed policy and both environment/action seeds should reproduce a trajectory','Identical 946-step traces; score -21'],
],[1.15,2.7,1.65])
p('Baseline variance check','heading')
p('With a frozen policy, estimate b from 20 separate pilot games. Compare gradients on paired fresh trajectories using the sum of component sample variances.','small')
v=checks['gradient_variance']['sum_of_component_gradient_variances']
vp=variance['statistics']
table([
    ['Game','Paired games','No baseline','EMA','EMA / none'],
    ['CartPole',checks['gradient_variance']['paired_episodes'],f'{v["none"]:.2f}',f'{v["historical_ema"]:.2f}',f'{checks["gradient_variance"]["variance_ratio_ema_to_none"]:.4f}'],
    ['Pong',variance['paired_episodes'],f'{vp["none"]["sum_component_variances"]:.2f}',f'{vp["historical_ema"]["sum_component_variances"]:.2f}',f'{variance["variance_ratio"]:.4f}'],
],[1,1.1,1.2,1.2,1.1])
p('The variance reduction is about 83% for CartPole and 82% for Pong at these policies; the Pong check uses a default-initialized policy. This does not guarantee higher scores. All trained settings use one seed, and the follow-up does not isolate standardization from outer-weight removal.','small')
p('Edge cases: preprocessing preserves raw frames; frame memory resets each game; both done flags are handled. Log probabilities use logits and non-finite losses are rejected. The saved replay starts with 10 zero rewards before any point is scored.','small')
p('With three more months','heading')
p('Repeat matched-budget ablations across at least five seeds; test value baselines and gradient variance at trained checkpoints. Use fresh test seeds for selected models. For the shuttle, vary arrivals, capacity and cap rules, checking deterministic cases against exact solutions.','small')

page('8. Code and reproduction')
p('GitHub: jingliangguo2027 / mlds-assignment-1<br/><link href="https://github.com/jingliangguo2027/mlds-assignment-1" color="#173957">https://github.com/jingliangguo2027/mlds-assignment-1</link>','small')
p('From hw1, install a suitable PyTorch build, then:')
code('python -m pip install -r requirements.txt\npython main.py\npython value_iteration.py\npython policy_iteration/main.py\npython policy_iteration/validate.py --experiment all')
p('Original experiments','heading')
p('From policy_gradient, replace BASELINE with none, then ema. Use new output folders.','small')
code('python train.py --game cartpole --baseline BASELINE \\\n  --episodes 2000 --eval-episodes 500 --lr 0.01 \\\n  --output new_cartpole_BASELINE\nCUDA_VISIBLE_DEVICES=0 python train.py --game pong \\\n  --baseline BASELINE --episodes 5000 --eval-episodes 500 \\\n  --hidden 128 --initialization he_small_output --lr 0.001 \\\n  --device cuda --output new_pong_BASELINE')
p('Pong follow-up','heading')
code('CUDA_VISIBLE_DEVICES=0 python pong_standardized.py \\\n  --episodes 4160 --device cuda --output new_pong_standardized')
p('The saved training curves use the first 4000 games; the final test uses the checkpoint at 4160. The command above reproduces the final endpoint; evaluate its checkpoint with:','small')
code('CUDA_VISIBLE_DEVICES=0 python pong_standardized.py \\\n  --evaluate-checkpoint new_pong_standardized/checkpoint.pt \\\n  --eval-episodes 500 --vector-envs 8 --device cuda \\\n  --output new_pong_test')
p('Redraw recorded figures with --plot-only --output results/pong_standardized.','small')
p('Validation and report','heading')
code('# From policy_gradient:\npython validate.py\nCUDA_VISIBLE_DEVICES=0 python validate.py --extra pong-variance --device cuda\nCUDA_VISIBLE_DEVICES=0 python validate.py --extra chunk --device cuda\n# From hw1:\npython build_report.py')
p('Default validation checks exact gradients, Pong preprocessing/replay and CartPole gradient variance. Extra modes check Pong variance and chunked gradients. They print fresh results without overwriting saved evidence.','small')
p('Scores and settings: policy_gradient/results/. Validation and tuning evidence: policy_gradient/evidence.json. New curves use results/pong_standardized/metrics.json; the new test is in evaluation.json. Original test seeds: 1000000-1000499; tuning: 200000-200049; new test: 9000000-9000499, with independent per-game action RNGs.','small')
p('Original Pong runners use action seed 10000+episode and environment seed episode. PyTorch 2.14.1 (+cu126 for Pong), Gymnasium 1.4.0, ALE 0.12.1; Pong uses one RTX A6000. Exact trajectories can vary across runtime versions. Historical scripts remain in <link href="https://github.com/jingliangguo2027/mlds-assignment-1/tree/e8ec206580c0c26dac98153dbd1bf1438fd36f4b/hw1" color="#173957">commit e8ec206</link>.','small')

def footer(canvas,doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#53616d'))
    canvas.drawRightString(A4[0]-44,22,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=44,rightMargin=44,
                      topMargin=40,bottomMargin=44,title='assignment1',author='Jingliang Guo')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
