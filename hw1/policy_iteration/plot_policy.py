"""Plot the dispatch policy."""
from pathlib import Path


def plot_policy(states, policy, gamma, output_folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.step(states, policy, where="post", color="#2563eb", linewidth=2)
    ax.set(
        title="Shuttle dispatch: optimal policy (policy iteration)",
        xlabel="Number of waiting passengers",
        ylabel="Optimal action",
        xlim=(states[0], states[-1]),
        ylim=(-0.15, 1.25),
        yticks=[0, 1],
        yticklabels=["Wait (0)", "Dispatch (1)"],
    )
    ax.text(0.03, 0.95, f"Discount factor = {gamma}", transform=ax.transAxes, va="top")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    output_path = Path(output_folder) / "optimal_policy.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
