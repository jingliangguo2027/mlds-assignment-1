"""Rebuild an honest tuning report from completed, recorded validation blocks."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'tuning_results')
root=parser.parse_args().output
audit=json.loads((root/'audit.json').read_text())
reference=audit['initial_validation']
initial_score=float(np.mean([r['score'] for r in reference]))
initial_j=float(np.mean([r['discounted_return'] for r in reference]))
order=['lr_0001','lr_001','lr_01','batch10_lr001','ema_lr001']
paths=sorted(root.glob('*/block_*.json'),key=lambda p:(order.index(p.parent.name) if p.parent.name in order else len(order),p.parent.name,int(p.stem.split('_')[1])))
rows=[]
lines=['# Pong：结果、分析与下一步','','## 实验规则','',
       '保留原公式及 gamma=0.99。每组从相同初始网络开始，固定训练种子；每次最多增加100局。',
       '每次用相同20局验证。继续标准：比上一检查点（首次为初始模型）平均增加至少1分，且配对bootstrap区间下界大于0。',
       '只有一组训练种子；这些区间只描述验证局波动，不代表跨训练种子的可靠性。反复使用的验证集只能用于调参，最终500局测试另外保留。',
       '短期没有改善只能支持停止这次短期筛选，不能证明该配置经过更长训练仍然无效。','',
       '## 更新过程检查','',
       f'- 分块梯度与整段梯度相对误差：{audit["chunk_vs_full_gradient_relative_error"]:.3g}。',
       f'- 梯度范数：{audit["gradient_norm"]:.3f}，非零且有限。',
       f'- 在同一轨迹上沿负梯度小步更新，代理损失从 {audit["surrogate_before"]:.6f} 降到 {audit["surrogate_after_descent"]:.6f}；这不等于真实游戏表现必然改善。',
       f'- 未训练网络：验证总分 {initial_score:.2f}，折扣目标 {initial_j:.3f}。',
       f'- 旧1380局模型：验证总分 {audit["old_vs_initial"]["mean"]:.2f}，折扣目标 {audit["old_vs_initial"]["mean_discounted_return"]:.3f}，无明确得分提升。','',
       '## 逐项记录','']
for path in paths:
    r=json.loads(path.read_text()); c=r['config']; v=r['vs_initial']; u=r['updates_last_block']
    label=path.parent.name+f' / {r["episodes"]}'
    rows.append((label,v['mean'],v['mean_discounted_return'],v['mean_entropy_first256']))
    comparison=r['vs_previous_block'] or v
    interval=comparison['paired_bootstrap_95_interval']
    if v['mean_saturated_first256']>.5:
        analysis='动作过早趋于固定：抽查状态中超过一半的最大动作概率大于99%；较大更新损害了探索。'
    elif comparison['clear_gain']:
        analysis='在当前固定验证集上达到预设改善门槛；需要继续100局检查持续性，并用新验证种子确认。'
    else:
        analysis='未达到预设改善门槛；有更新，但本次短期验证未显示可靠的得分改善。'
    decision='继续100局，再次比较' if comparison['clear_gain'] else '停止此配置；分析后再更改一个因素'
    followups={
        'lr_0001':'动作概率变化很小，下一实验只把学习率提高到0.001。',
        'lr_001':'仍未达门槛，下一实验只把学习率提高到作业默认0.01。',
        'lr_01':'发现过早固定动作，停止放大学习率；先测量梯度方差，再用0.001试累积10局更新。',
        'batch10_lr001':'改善仅0.10分，停止此配置；另从0.001单局更新对照出发加入EMA baseline。',
        'ema_lr001':'降噪没有转化为得分提升，停止此配置。本轮五组短期筛选结束，没有配置达到自动续训门槛。'
    }
    lines += [f'### {label}','',
              f'设置：学习率 {c["lr"]}，每 {c["batch_size"]} 局更新，baseline={c["baseline"]}。',
              f'**结果**：20局验证均分 {v["mean"]:.2f}；折扣目标 {v["mean_discounted_return"]:.3f}；相对比较对象得分变化 {comparison["paired_score_gain"]:+.2f}，配对95% bootstrap区间 [{interval[0]:+.2f}, {interval[1]:+.2f}]。',
              f'训练前50局均分 {r["training_first50"]:.2f}，最近50局 {r["training_last50"]:.2f}。',
              f'平均梯度范数 {np.mean([x["gradient_norm"] for x in u]):.3f}；每次参数相对变化 {np.mean([x["relative_parameter_delta"] for x in u]):.5f}；抽查状态动作熵 {v["mean_entropy_first256"]:.3f}（两动作均匀时约0.693）。',
              f'**结果分析**：{analysis}',f'**是否优化**：{decision}。',
              f'**后续动作**：{followups.get(path.parent.name,"根据当前块结果决定下一步。") if r["episodes"]==100 else decision}','']
variance=root/'gradient_variance.json'
if variance.exists():
    v=json.loads(variance.read_text())
    lines += ['## 梯度方差诊断','',
              '先用独立20局估计历史EMA参考值，然后在另外20局相同轨迹上成对计算有/无baseline的梯度。策略冻结。',
              f'方差分量之和：无baseline {v["statistics"]["none"]["sum_component_variances"]:.3f}；有baseline {v["statistics"]["historical_ema"]["sum_component_variances"]:.3f}；比值 {v["variance_ratio"]:.3f}。',
              '这支持尝试baseline，但不保证训练得分会改善。baseline实验属于作业Part 2；Part 1的无baseline实验独立保留。','']
lines += ['## 本轮结论与限制','',
          '没有发现梯度被切断、参数没有更新或分块梯度计算错误；这不等于证明所有实现细节都正确。',
          '学习率0.01出现明显动作饱和。历史baseline减少了固定策略下的梯度方差，但没有保证训练效果。',
          '部分配置的折扣目标变好而总分变差，因此不能用总分曲线单独判断优化器是否工作。两项指标都应持续报告。',
          '100局是本轮筛选预算，不是已证实足够的学习预算。逐100局无改善就淘汰的规则可能错过需要更长学习的配置；本轮无法确定长期最优超参数或解释全部失败原因。',
          '2026-10-24首轮五组实验均未触发继续条件，训练进程已停止；旧1380局断点及本轮各100局断点保留，最终500局测试尚未执行。','',
          '## 复现','',
          '服务器目录：`/nfs/home/nxt1510/hw1_deepdish_20261024/hw1/policy_gradient`；Deepdish4，一张A6000，myenv。',
          '完整配置、原始20局验证分数、每次梯度及参数变化保存在各实验目录的JSON和checkpoint.pt中。',
          '在policy_gradient目录执行（须有可用的单张GPU）。以下使用新输出目录，保留原实验结果：','',
          '```sh','python validate_tuning.py',
          'CUDA_VISIBLE_DEVICES=0 python tune_pong.py --output tuning_repro --audit --old-checkpoint results/pong_none/checkpoint.pt',
          'CUDA_VISIBLE_DEVICES=0 python tune_pong.py --output tuning_repro --name lr_001 --lr 0.001 --target 100',
          '# 仅当上一块达到改善门槛时，才把同一配置target增加到200。',
          '# 改学习率、batch-size或baseline必须使用新的name，从头对照。',
          'CUDA_VISIBLE_DEVICES=0 python diagnose_pong_variance.py --output tuning_repro','python summarize_tuning.py --output tuning_repro','```','']
(root/'实验分析.md').write_text('\n'.join(lines)+'\n')
if rows:
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,index,title,ref in zip(axes,(1,2,3),('Validation score (higher is better)','Discounted objective','Action entropy (first 256 states)'),(initial_score,initial_j,.693147)):
        ax.scatter([r[index] for r in rows],range(len(rows)),color='#3c7fa1',s=40)
        ax.set_yticks(range(len(rows)),[r[0] for r in rows] if index==1 else ['']*len(rows))
        ax.axvline(ref,color='#cc6a37',linestyle='--',label='Initial / uniform entropy')
        ax.set_title(title,fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(root/'tuning_comparison.png',dpi=180)
    plt.close(fig)
print(root/'实验分析.md')
