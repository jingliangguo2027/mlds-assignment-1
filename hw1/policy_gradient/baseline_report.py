"""Compare the two fixed-setting 5000-episode runs; never train or evaluate."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import load_checkpoint, make_policy, policy_fingerprint, save_json
from tune_pong import comparison

root = Path(__file__).resolve().parent / 'tuning_results'
names = ['width128_he_small', 'width128_he_small_ema']
labels = ['No baseline', 'EMA baseline']
runs, checkpoints, stats = [], [], []
for name in names:
    folder = root / name
    run = json.loads((folder / 'block_5000.json').read_text())
    checkpoint = load_checkpoint(folder / 'checkpoint_5000.pt')
    assert run['episodes'] == len(checkpoint['records']) == 5000
    assert run['config'] == checkpoint['config']
    model = make_policy('pong', 128, 'he_small_output')
    model.load_state_dict(checkpoint['policy'])
    assert policy_fingerprint(model) == run['model_fingerprint']
    assert all(torch.isfinite(v).all() for v in checkpoint['policy'].values())
    assert [r['seed'] for r in run['validation']] == list(range(200000, 200050))
    scores = np.array([r['score'] for r in run['validation']])
    stats.append(dict(experiment=name, mean=float(scores.mean()), std=float(scores.std()),
        wins=int((scores > 0).sum()), min=float(scores.min()), max=float(scores.max()),
        discounted_return=run['vs_initial']['mean_discounted_return'],
        entropy=run['vs_initial']['mean_entropy_first256'],
        saturation=run['vs_initial']['mean_saturated_first256'],
        training_last100=float(np.mean([r['score'] for r in checkpoint['records'][-100:]])),
        baseline_final=checkpoint['baseline_value'],
        baseline_used_range=[min(r['baseline_used'] for r in checkpoint['records']),
                             max(r['baseline_used'] for r in checkpoint['records'])],
        update_last100={key: float(np.mean([u[key] for u in checkpoint['updates'][-100:]]))
            for key in ['gradient_norm', 'relative_parameter_delta', 'probe_kl']},
        fingerprint=run['model_fingerprint'],
        checkpoint_sha256=hashlib.sha256((folder / 'checkpoint_5000.pt').read_bytes()).hexdigest()))
    runs.append(run)
    checkpoints.append(checkpoint)
assert runs[0]['initial_model_fingerprint'] == runs[1]['initial_model_fingerprint']
assert runs[0]['config']['baseline'] == 'none' and runs[1]['config']['baseline'] == 'ema'
assert all(runs[0]['config'][k] == runs[1]['config'][k] for k in runs[0]['config'] if k != 'baseline')
assert checkpoints[1]['records'][0]['baseline_used'] == 0
paired = comparison(runs[1]['validation'], runs[0]['validation'])
windows = []
for end in range(500, 5001, 500):
    windows.append(dict(start=end-499, end=end, means=[float(np.mean(
        [r['score'] for r in c['records'][end-500:end]])) for c in checkpoints]))
summary = dict(status='complete', episodes=5000, validation_episodes=50,
    final500_test=False, completed_utc=runs[1]['completed_utc'],
    runs=stats, paired_baseline_minus_none=paired, training_windows=windows,
    initial_model_fingerprint=runs[0]['initial_model_fingerprint'],
    baseline_definition='Use historical b, then b=0.9*b+0.1*time-weighted mean(G); initially0.',
    limitation='Single training seed; repeated tuning seeds. Intervals cover validation variation, not training-seed uncertainty.')
summary['review'] = dict(time_utc=runs[1]['completed_utc'],
    stage='EMA baseline与无baseline的5000局对照完成', experiment=names[1], episodes=5000,
    result=f'相同50局冻结调参验证：无baseline均分{stats[0]["mean"]:.2f}，EMA均分{stats[1]["mean"]:.2f}，差{paired["paired_score_gain"]:+.2f}，配对95%区间[-0.76,1.20]。得分标准差分别2.43、3.26，两组均0胜；折扣回报分别-0.335、-0.235，差+0.100，区间[-0.014,0.219]。',
    analysis='EMA在中期更早改善：2001–2500局均分-19.36对-20.72，3001–3500局-18.72对-20.16；4501–5000局已接近，为-18.78对-18.82。最终验证两项差值区间都跨0，不能确认5000局终点的稳定优势。EMA前256步动作熵0.094低于0.223，概率>99%的比例73.43%高于40.53%，探索更少。更新幅度更小不能单独证明梯度方差降低；结果也不能证明baseline普遍无效。',
    decision='按用户预算完成两组各5000局；保存模型、Adam状态、日志、曲线与分析，结束本轮GPU任务和复查，不自动追加训练或更改设置。',
    reason='固定预算对照已完成。中期改善没有转化为明确的终点得分优势；本次只有一个训练种子，50局也反复用于调参。等待用户讨论后续，正式独立500局测试仍未运行。')
save_json(root / 'baseline_5000_summary.json', summary)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for c, label in zip(checkpoints, labels):
    for ax, key in zip(axes[0], ['score', 'discounted_return']):
        values = [r[key] for r in c['records']]
        ax.plot(np.arange(100, 5001), np.convolve(values, np.ones(100)/100, mode='valid'), label=label)
        ax.set_xlabel('Training episode'); ax.grid(alpha=.2); ax.legend(fontsize=8)
axes[0, 0].set_title('Game score: 100-episode moving average')
axes[0, 1].set_title('Discounted return: 100-episode moving average')
axes[1, 0].plot(np.arange(1, 5001), [r['baseline_used'] for r in checkpoints[1]['records']], lw=.8)
axes[1, 0].axhline(0, color='gray', ls='--', label='No baseline')
axes[1, 0].set(title='Historical EMA baseline used in each update', xlabel='Training episode', ylabel='b')
axes[1, 0].grid(alpha=.2)
for run, label in zip(runs, labels):
    axes[1, 1].hist([r['score'] for r in run['validation']], bins=np.arange(-21.5, 1.5),
                    histtype='step', lw=1.8, label=label)
axes[1, 1].set(title='50 frozen tuning games (not final 500 test)', xlabel='Game score', ylabel='Count')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=.2)
fig.savefig(root / 'baseline_5000_comparison.png', dpi=180)
plt.close(fig)
lines = ['# Pong：baseline固定5000局对照', '',
    '仅将baseline从none改为历史EMA；初始权重、seed、128单元、He-small初始化、lr0.001、batch1、内外层gamma0.99、动作2/3及奖励均一致。两组均累计训练5000局。无baseline组曾在300局保存并完整继承Adam续训；EMA组从零训练，未继承旧模型。', '',
    '| 配置 | 验证均分 | 得分标准差（ddof=0） | 折扣回报 | 胜局/50 | 最后100局训练均分 |',
    '|---|---:|---:|---:|---:|---:|']
for label, s in zip(labels, stats):
    lines.append(f'| {label} | {s["mean"]:.2f} | {s["std"]:.3f} | {s["discounted_return"]:.3f} | {s["wins"]} | {s["training_last100"]:.2f} |')
lines += ['', f'EMA减去无baseline：得分变化{paired["paired_score_gain"]:+.2f}，配对bootstrap95%区间{paired["paired_bootstrap_95_interval"]}；折扣回报变化{paired["discounted_gain"]:+.3f}，区间{paired["discounted_gain_95_interval"]}。',
    '两组使用相同50个验证种子。区间仅反映这两个固定模型在验证局上的波动，不包含不同训练种子的变化；反复用于调参，不是正式独立500局测试。', '',
    '| 训练区间 | 无baseline均分 | EMA均分 |', '|---|---:|---:|']
for w in windows:
    lines.append(f'| {w["start"]}–{w["end"]} | {w["means"][0]:.2f} | {w["means"][1]:.2f} |')
lines += ['', '探索诊断仅统计每局前256步；更新诊断取最后100次更新，不代表整个训练的梯度方差。']
for label, s in zip(labels, stats):
    lines.append(f'{label}：验证动作熵{s["entropy"]:.3f}，动作概率>99%的比例{s["saturation"]:.2%}；最后100次更新平均梯度范数{s["update_last100"]["gradient_norm"]:.3f}，相对参数更新量{s["update_last100"]["relative_parameter_delta"]:.5f}。')
lines += [f'EMA最终b={stats[1]["baseline_final"]:.6f}，训练所用b范围{stats[1]["baseline_used_range"]}。', '',
          '## 分析与决定', '', summary['review']['analysis'], '', summary['review']['decision'], '', summary['review']['reason'], '',
          '![训练、baseline与冻结验证对照](baseline_5000_comparison.png)', '']
(root / 'baseline_5000_comparison.md').write_text('\n'.join(lines))
print(json.dumps(summary, ensure_ascii=False, indent=2))
