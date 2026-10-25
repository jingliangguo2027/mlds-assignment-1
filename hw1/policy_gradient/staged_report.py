"""Summarize staged experiments and recorded decisions, without imposing a hard stop."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import load_checkpoint

root=Path(__file__).resolve().parent/'tuning_results'
decision_path=root/'stage_decisions.json'
decisions=json.loads(decision_path.read_text()) if decision_path.exists() else []
known={(d['stage'],d.get('experiment','')) for d in decisions}
for path in root.glob('*/stage_decision.json'):
    d=json.loads(path.read_text())
    if (d['stage'],d.get('experiment','')) not in known:
        decisions.append(d)
decisions.sort(key=lambda d:d['time_utc'])
files=list(root.glob('*/block_*.json'))
experiments={}
for p in files:
    r=json.loads(p.read_text())
    experiments.setdefault(p.parent.name,[]).append(r)
for rows in experiments.values():
    rows.sort(key=lambda r:r['episodes'])
lines=['# Pong 分阶段优化记录','',
       '当前网络宽度与初始化对照恢复原公式 `-sum(gamma**t * (G_t-b) * log pi)`，学习率0.001，G内部gamma=0.99。历史no_outer_lr001曾去掉外层权重；其900局续训在640局停止，未完成的640局不能当作900局结果。',
       '去掉外层权重改变了梯度估计；新损失不再标注为原始整局折扣目标的严格等价梯度。整局得分和折扣回报仍同时作为对照指标。',
       '首轮100局用于粗筛，之后通常每300局检查；首次将100局补到累计300局。',
       '续训依据是多个阶段的得分、折扣目标、探索情况和更新幅度，不再执行100局未提高1分就淘汰的硬规则。',
       '从零训练使用相同训练种子；相同结构及初始化方案使用相同初始模型，宽度实验只保持初始化方法和种子一致，各结构单独建立未训练基准。学习率分支继承指定模型和Adam状态。首轮验证20局，后续50局；阶段比较只使用共有的验证种子。',
       '验证用于调参，最终500局测试种子保持独立。单训练种子和反复验证均限制结论，不能据此宣称长期最优。','',
       '## 阶段结果','',
       '| 配置 | 隐藏单元 | 初始化 | Baseline | 外层折扣 | 累计训练局数 | 本次新增 | 验证局数 | 游戏得分 | 折扣回报 | 动作熵 | 动作概率>99%的比例 |',
       '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
summary_path=root/'architecture_summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text())
    current=['','## 历史300局网络对照结论','',summary['conclusion'],'',
             '| 隐藏单元 | 初始化 | 300局后的游戏得分 | 折扣回报 |',
             '|---:|---|---:|---:|']
    for row in summary['runs']:
        v=row['vs_128_default']
        current.append(f'| {row["hidden"]} | {row["initialization"]} | {v["mean"]:.2f} | {v["mean_discounted_return"]:.3f} |')
    current+=['',summary['decision'],
              '以上均为同50局调参验证，128默认使用历史300局控制；最终500局测试尚未完成。',
              '区间只反映验证局波动，尚未检验不同训练种子的稳定性。',
              '', '![相对128默认的得分变化](architecture_comparison.png)','']
    lines[1:1]=current
long_path=root/'long_run_5000_summary.json'
if long_path.exists():
    long_summary=json.loads(long_path.read_text())
    review=long_summary['review']
    current=['','## 无baseline：固定设置5000局训练完成','',
             '128隐藏单元、He隐藏层与小输出层初始化、学习率0.001、无baseline；保留内外层折扣。从300局模型及Adam状态继续4700局，中途没有改参数。','',
             review['result'],'',review['analysis'],'',review['decision'],'',review['reason'],'',
             '| 训练区间 | 整局得分均值 | 折扣回报均值 |',
             '|---|---:|---:|']
    for window in long_summary['training_windows']:
        current.append(f'| {window["start"]}–{window["end"]} | {window["mean_score"]:.2f} | {window["mean_discounted_return"]:.3f} |')
    current+=['','![5000局训练趋势](long_run_5000_training.png)','']
    lines[1:1]=current
baseline_path=root/'baseline_5000_summary.json'
if baseline_path.exists():
    baseline_summary=json.loads(baseline_path.read_text())
    review=baseline_summary['review']
    lines[1:1]=['','## 最新：baseline的5000局对照完成','',review['result'],'',review['analysis'],'',
                review['decision'],'',review['reason'],'',
                '[详细统计与对照方法](baseline_5000_comparison.md)','',
                '![baseline对照](baseline_5000_comparison.png)','']
final_path=root.parent/'final_test_comparison.json'
if final_path.exists():
    final=json.loads(final_path.read_text())
    none,ema=final['models']['none'],final['models']['ema']
    lines[1:1]=['','## 最新：正式500局测试已完成','',
                f'无baseline：均值{none["mean"]:.3f}，标准差{none["std"]:.3f}；EMA：均值{ema["mean"]:.3f}，标准差{ema["std"]:.3f}。',
                f'EMA减去无baseline的得分差{final["mean_difference_ema_minus_none"]:+.3f}，配对95%区间{final["paired_bootstrap_95_interval"]}。',
                '两组各500局，使用1000000–1000499独立测试种子；模型冻结。完整直方图、统计和中文分析见[正式实验报告](../report_results.md)。',
                '下文保留各阶段当时的判断；其中“正式测试尚未运行”等表述为历史状态，本段为最新状态。','']
for name,rows in sorted(experiments.items()):
    for r in rows:
        v=r['vs_initial']
        outer='去掉' if r['config'].get('omit_time_discount',False) else '保留'
        lines.append(f'| {name} | {r["config"].get("hidden",128)} | {r["config"].get("initialization","default")} | {r["config"].get("baseline","none")} | {outer} | {r["episodes"]} | {r.get("new_training_episodes",100)} | {len(r["validation"])} | {v["mean"]:.2f} | {v["mean_discounted_return"]:.3f} | {v["mean_entropy_first256"]:.3f} | {100*v["mean_saturated_first256"]:.1f}% |')
for name in sorted(experiments):
    lineage_path=root/name/'lineage.json'
    if lineage_path.exists():
        d=json.loads(lineage_path.read_text())
        lines+=['',f'{name} 继承 {d["source_experiment"]} 的 {d["episodes_at_fork"]} 局模型与Adam状态；学习率从 {d["previous_lr"]} 改为 {d["new_lr"]}。累计局数包含继承部分，并非全部使用新学习率。']
lines+=['','## 结果、分析与调整理由','']
for d in decisions:
    lines += [f'### {d["stage"]}','', f'记录时间（UTC）：{d["time_utc"]}',
              f'**结果**：{d["result"]}',f'**分析**：{d["analysis"]}',
              f'**决定**：{d["decision"]}',f'**调整理由**：{d["reason"]}','']
lines+=['## 相邻检查点对照','']
for name,rows in sorted(experiments.items()):
    for r in rows:
        p=r.get('vs_previous_block')
        if p is None: continue
        lines += [f'- {name}，累计{r["episodes"]}局：共有验证种子{p.get("paired_episodes",20)}个；得分变化 {p["paired_score_gain"]:+.2f}，配对bootstrap区间 {p["paired_bootstrap_95_interval"]}；折扣目标变化 {p.get("discounted_gain","旧记录未计算")}。']
confirmations=list(root.glob('confirmation_*.json'))
if confirmations:
    lines+=['','## 新种子复核（仍非最终测试）','']
    if (root/'validation_seed_usage.json').exists():
        next_seed=json.loads((root/'validation_seed_usage.json').read_text())['next_unused_confirmation_start']
        lines+=[f'种子核对：300000–300019此前用于初始策略的baseline估计，并非全新种子。下一次未使用的复核种子从{next_seed}开始，完整使用范围见validation_seed_usage.json。','']
    for path in sorted(confirmations):
        r=json.loads(path.read_text())
        if r.get('status')!='complete':
            lines+=[f'{path.name}：尚未完成，不能将部分结果当作完整复核。','']
            continue
        lines += [f'{path.name}：{r["episodes"]}局，起始种子{r["seed_start"]}。', '',
                  '| 模型 | 得分 | 折扣回报 | 相对未训练模型的折扣增益及区间 |',
                  '|---|---:|---:|---|']
        ref=r['reference']
        lines += [f'| 未训练 | {np.mean([x["score"] for x in ref]):.2f} | {np.mean([x["discounted_return"] for x in ref]):.3f} | — |']
        for candidate in r['candidates']:
            v=candidate['vs_initial']
            lines += [f'| {candidate["checkpoint"]} | {v["mean"]:.2f} | {v["mean_discounted_return"]:.3f} | {v["discounted_gain"]:+.3f}，{v["discounted_gain_95_interval"]} |']
        lines+=['']
lines+=['','## 实现与复现','',
        'train.py 支持两种损失，默认保留旧行为；--omit-time-discount 启用用户要求的新变体，且禁止同名检查点混用两种损失。动作和G计算不变。',
        '更新算法的解析梯度、分块与整段梯度、批次平均梯度已验证，原始证据在 audit.json、validation_checks.json 中。',
        '命令例子（仅在对应训练预算尚未达到时运行）：','',
        '```sh','CUDA_VISIBLE_DEVICES=0 python tune_pong.py --name lr_001 --lr 0.001 --target 300',
        '# 根据阶段结果决定是否继续，下一检查点示例：',
        'CUDA_VISIBLE_DEVICES=0 python tune_pong.py --name lr_001 --lr 0.001 --target 600',
        '# 用户要求的新对照，从零初始化，不继承旧模型：',
        'CUDA_VISIBLE_DEVICES=0 python tune_pong.py --name no_outer_lr001 --lr 0.001 --target 300 --omit-time-discount',
        'python staged_report.py','```','',
        '同名实验只允许续训，不允许静默更改学习率、batch或baseline。改变配置使用新名字；fork_tuning.py 只改变学习率，lineage.json 记录模型来源，validation_at_fork.json 保存继承模型的既有验证。',
        '每个阶段保存 checkpoint_<局数>.pt，原来的100局、1380局模型均保留。最终正式测试仍未完成。']
(root/'分阶段实验分析.md').write_text('\n'.join(lines)+'\n')
# Training curves use raw scores, not the tuning validation samples.
selected=[name for name,rows in experiments.items() if rows[-1]['episodes']>=300]
if selected:
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for name in sorted(selected):
        last=experiments[name][-1]['episodes']
        c=load_checkpoint(root/name/f'checkpoint_{last}.pt')
        records=c['records']
        lineage_path=root/name/'lineage.json'
        first=json.loads(lineage_path.read_text())['episodes_at_fork'] if lineage_path.exists() else 100
        for ax,key in zip(axes,('score','discounted_return')):
            values=np.array([r[key] for r in records])
            smooth=np.convolve(values,np.ones(100)/100,mode='valid')
            x=np.arange(100,len(values)+1)
            ax.plot(x[x>=first],smooth[x>=first],label=name)
            ax.set_xlabel('Training episode'); ax.grid(alpha=.2)
    axes[0].set_title('Game score: 100-episode average')
    axes[1].set_title('Discounted return: 100-episode average')
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(root/'staged_training.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in sorted(selected):
        rows=[r for r in experiments[name] if len(r['validation'])==50]
        fork_path=root/name/'validation_at_fork.json'
        if fork_path.exists():
            rows=[json.loads(fork_path.read_text())]+rows
        for ax,key in zip(axes,('mean','mean_discounted_return')):
            ax.plot([r['episodes'] for r in rows],[r['vs_initial'][key] for r in rows],marker='o',label=name)
    reference=json.loads((root/'initial_validation_50.json').read_text())
    for ax,key in zip(axes,('score','discounted_return')):
        ax.axhline(np.mean([r[key] for r in reference]),ls='--',color='gray',label='Untrained 128/default')
        ax.set_xlabel('Training episode');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_title('Validation game score (50 common seeds)')
    axes[1].set_title('Validation discounted return (higher is better)')
    fig.savefig(root/'staged_validation.png',dpi=180);plt.close(fig)
if summary_path.exists():
    rows=[r for r in summary['runs'] if r['experiment']!='lr_001']
    gains=np.array([r['vs_128_default']['paired_score_gain'] for r in rows])
    bounds=np.array([r['vs_128_default']['paired_bootstrap_95_interval'] for r in rows])
    labels=[f'{r["hidden"]} / '+('default' if r['initialization']=='default' else 'He + small output') for r in rows]
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    ax.errorbar(gains,np.arange(len(rows)),xerr=np.stack([gains-bounds[:,0],bounds[:,1]-gains]),fmt='o',capsize=5,color='#315b91')
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(np.arange(len(rows)),labels)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_title('300 episodes: score difference vs. 128 / default')
    ax.set_xlabel('Mean score difference; paired 95% bootstrap interval (50 tuning games)')
    fig.savefig(root/'architecture_comparison.png',dpi=180);plt.close(fig)
print(root/'分阶段实验分析.md')
if long_path.exists():
    c=load_checkpoint(root/'width128_he_small'/'checkpoint_5000.pt')
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    for ax,key,title in zip(axes,('score','discounted_return'),('Game score','Discounted return')):
        values=np.array([r[key] for r in c['records']])
        ax.plot(np.arange(1,len(values)+1),values,color='#8099ae',alpha=.22,lw=.5,label='Each training episode')
        ax.plot(np.arange(100,len(values)+1),np.convolve(values,np.ones(100)/100,mode='valid'),color='#145b89',lw=1.7,label='100-episode moving average')
        ax.axvline(300,color='#a05a24',ls='--',lw=1,label='Resumed from episode 300')
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    axes[0].set_title('Fixed settings: 128 units, He + small output, lr=0.001, no baseline')
    axes[-1].set_xlabel('Training episode (outer time discount retained)')
    fig.savefig(root/'long_run_5000_training.png',dpi=180);plt.close(fig)
