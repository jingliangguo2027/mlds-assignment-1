"""画图独立存放，学习算法时看 train.py。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(training_scores, test_scores, output_folder):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(range(1, len(training_scores) + 1), training_scores, alpha=0.4, label="Episode reward")
    if len(training_scores) >= 100:
        average = np.convolve(training_scores, np.ones(100) / 100, mode="valid")
        ax.plot(range(100, len(training_scores) + 1), average, color="#dc2626", label="100-episode average")
    ax.set(title="CartPole-v1: REINFORCE without baseline", xlabel="Training episode", ylabel="Episode reward", ylim=(0, 520))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output_folder / "training_rewards.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.hist(test_scores, bins=np.linspace(0, 500, 26), color="#2563eb", edgecolor="white")
    ax.set(title="CartPole-v1: 500 evaluation episodes", xlabel="Episode reward", ylabel="Number of episodes", xlim=(0, 500))
    ax.text(0.03, 0.95, f"Mean = {np.mean(test_scores):.2f}\nStd = {np.std(test_scores):.2f}", transform=ax.transAxes, va="top")
    fig.savefig(output_folder / "test_rewards.png", dpi=200)
    plt.close(fig)
