"""Read actual experiment results, compare baselines, and prepare report tables/figures."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_completed(folder):
    path = folder / "metrics.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    config = json.loads((folder / "config.json").read_text())
    status_path = folder / "status.json"
    if status_path.exists() and json.loads(status_path.read_text())["status"] != "complete":
        return None
    if len(result["training_scores"]) != config["episodes"] or len(result["test_scores"]) != 500:
        return None
    if any(result["config"][key] != config[key] for key in ("game", "baseline", "seed", "lr", "episodes")):
        return None
    # 从原始分数重新计算，不只信任保存的汇总数。
    assert np.isclose(np.mean(result["test_scores"]), result["test_mean"])
    assert np.isclose(np.std(result["test_scores"]), result["test_std"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    collected, missing = {}, []
    lines = ["# Policy-gradient experimental results", "",
             "Generated from recorded scores. Missing experiments remain PENDING.", "",
             "| Game | Baseline | Training episodes | Test episodes | Mean | Std | Device |",
             "|---|---|---:|---:|---:|---:|---|"]
    for game in ("cartpole", "pong"):
        for baseline in ("none", "ema"):
            key = f"{game}_{baseline}"
            folder = ROOT / "results" / key
            final_folder = ROOT / "results" / f"pong_final_{baseline}"
            if game == "pong" and final_folder.exists():
                folder = final_folder
            result = load_completed(folder)
            if result is None:
                missing.append(key)
                lines.append(f"| {game} | {baseline} | PENDING | PENDING | PENDING | PENDING | PENDING |")
            else:
                collected[key] = result
                lines.append(f"| {game} | {baseline} | {len(result['training_scores'])} | 500 | {result['test_mean']:.3f} | {result['test_std']:.3f} | {result['config']['device']} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for game in ("cartpole", "pong"):
        keys = [f"{game}_none", f"{game}_ema"]
        if not all(key in collected for key in keys):
            continue
        a, b = [collected[key] for key in keys]
        for key in ("seed", "episodes", "lr", "gamma", "game"):
            assert a["config"][key] == b["config"][key], f"Unmatched comparison setting: {key}"
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for key, label in zip(keys, ("No baseline", "Historical EMA baseline")):
            scores = collected[key]["training_scores"]
            if len(scores) >= 100:
                ax.plot(range(100, len(scores)+1), np.convolve(scores, np.ones(100)/100, mode="valid"), label=label)
        ax.set(title=f"{game}: baseline comparison, same seed and training budget", xlabel="Training episode", ylabel="100-episode mean reward")
        ax.legend()
        fig.savefig(ROOT / f"{game}_baseline_comparison.png", dpi=200)
        plt.close(fig)

    lines += ["", "## Interpretation", "",
              "The EMA baseline uses only preceding episodes and is updated after the actor step.",
              "Each reported model is frozen and evaluated for 500 stochastic-policy episodes.",
              "These runs use one seed; they do not establish variability across random seeds.",
              "A smaller policy-gradient variance does not imply a smaller standard deviation of episode scores.",
              "CPU results are local experiments and do not certify completion of the Lab's Deepdish/GPU setup."]
    if all(f'pong_{b}' in collected for b in ('none', 'ema')):
        a, b = [collected[f'pong_{v}'] for v in ('none', 'ema')]
        test_records = []
        for baseline, result in zip(('none', 'ema'), (a, b)):
            folder = ROOT / 'results' / f'pong_final_{baseline}'
            saved = json.loads((folder / 'evaluation_scores.json').read_text())
            assert saved['model_id'] == result['config']['model_id'] == result['model_id']
            assert saved['scores'] == result['test_scores']
            assert [r['seed'] for r in saved['rows']] == list(range(1000000, 1000500))
            test_records.append(saved['rows'])
            fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
            ax.hist(result['test_scores'], bins=np.arange(-21.5, 22.5, 1), edgecolor='white')
            ax.set(title=f'Pong: 500 final test games, baseline={baseline}',
                   xlabel='Episode reward', ylabel='Count')
            ax.text(.97, .95, f'Mean={result["test_mean"]:.3f}\nStd={result["test_std"]:.3f}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=.9))
            fig.savefig(folder / 'evaluation_rewards.png', dpi=200)
            plt.close(fig)
        delta = np.array(b['test_scores']) - np.array(a['test_scores'])
        rng = np.random.default_rng(42)
        interval = np.quantile(rng.choice(delta, (5000, 500), replace=True).mean(axis=1), [.025, .975])
        wins = [sum(score > 0 for score in r['test_scores']) for r in (a, b)]
        final_comparison = dict(test_episodes_per_model=500, std_ddof=0,
            seed_range=[1000000, 1000499], mean_difference_ema_minus_none=float(delta.mean()),
            paired_bootstrap_95_interval=interval.tolist(), wins=dict(none=wins[0], ema=wins[1]),
            models={label:dict(model_id=r['model_id'],mean=r['test_mean'],std=r['test_std'],
                              mean_discounted_return=r['test_mean_discounted_return'])
                    for label,r in zip(('none','ema'),(a,b))})
        (ROOT / 'final_test_comparison.json').write_text(json.dumps(final_comparison, indent=2)+'\n')
        conclusion = ('区间包含0，不能确认这两个模型的平均得分存在明确差异。'
                      if interval[0] <= 0 <= interval[1] else
                      '区间不包含0，在本次测试种子上两个固定模型的平均得分有明确差异。')
        lines += ['', '## Pong: final 500-game tests', '',
                  'Both selected models trained for 5000 episodes from the same initial weights: 128 hidden units, He hidden initialization and small output initialization, Adam lr=0.001, gamma=0.99, batch=1. The outer time discount was retained. Only the historical EMA baseline differs.',
                  'Each model was frozen and tested on seeds1000000–1000499, separate from tuning seeds200000–200049. Actions were sampled from the policy, with no parameter or baseline updates. Standard deviations use ddof=0.',
                  f'EMA minus no-baseline mean score: {b["test_mean"]-a["test_mean"]:+.3f}. These are final evaluation results for one training seed; they do not establish general superiority across training seeds.', '',
                  '### 中文结果分析', '',
                  f'无baseline：均值{a["test_mean"]:.3f}，标准差{a["test_std"]:.3f}，获胜{wins[0]}/500局。EMA baseline：均值{b["test_mean"]:.3f}，标准差{b["test_std"]:.3f}，获胜{wins[1]}/500局。',
                  f'EMA减去无baseline的均分差为{delta.mean():+.3f}，按相同测试种子配对的bootstrap95%区间为[{interval[0]:.3f}, {interval[1]:.3f}]。{conclusion}',
                  '这次是独立于调参的正式500局测试，不能与之前50局验证混用。区间反映固定模型的测试局波动，不包含重新训练时的随机性；单训练种子无法证明某种算法普遍更好。奖励标准差也不是策略梯度方差。',
                  '训练过程中EMA组在中期改善更早，但后期两组得分逐渐接近。最终模型均仍以输球为主；保留完整结果，不根据正式测试继续选择超参数。', '',
                  '![Pong no-baseline: final500 reward histogram](results/pong_final_none/evaluation_rewards.png)', '',
                  '![Pong EMA baseline: final500 reward histogram](results/pong_final_ema/evaluation_rewards.png)', '',
                  '![Pong training comparison](pong_baseline_comparison.png)', '',
                  '![Pong no-baseline: per-episode training rewards and MA100](results/pong_final_none/training_rewards.png)', '',
                  '![Pong EMA baseline: per-episode training rewards and MA100](results/pong_final_ema/training_rewards.png)']
    variance_path = ROOT / "validation_results.json"
    if variance_path.exists():
        variance = json.loads(variance_path.read_text())["gradient_variance"]
        lines += ["", "## Paired gradient-variance experiment", "",
                  f"Baseline estimated from {variance['pilot_episodes']} separate pilot episodes; paired gradients use {variance['paired_episodes']} further episodes under a fixed policy.",
                  f"Sum of component variances: {variance['sum_of_component_gradient_variances']}.",
                  f"EMA/no-baseline variance ratio: {variance['variance_ratio_ema_to_none']:.6f}."]
    (ROOT / "report_results.md").write_text("\n".join(lines)+"\n")
    summary = {"complete": not missing, "pending": missing,
               "experiments": {key: {"test_mean": value["test_mean"], "test_std": value["test_std"],
                                      "training_episodes": len(value["training_scores"]), "test_episodes": len(value["test_scores"])} for key, value in collected.items()}}
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("\n".join(lines[:10]))
    print("Pending:", missing)
    if args.require_complete and missing:
        raise SystemExit("Four completed 500-episode evaluations are required before final reporting")


if __name__ == "__main__":
    main()
