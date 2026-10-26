MAX_PEOPLE = 200
K = 15
GAMMA = 0.95
TOL = 1e-6

states = list(range(MAX_PEOPLE + 1))
actions = [0, 1]

def remaining_people(s, a):
    if a == 0:
        return s
    if a == 1:
        return max(s - K, 0)

def cost(s, a):
    return 100 * a + remaining_people(s, a) * 2

def preCost(s, values):
    totalCost = 0
    for k in range(1, 6):
        next_s = min(MAX_PEOPLE, s + k)
        totalCost += values[next_s][1]
    return totalCost / 5

V = [[0, 0.0] for _ in states]
iterations = 0
delta = float("inf")

while delta >= TOL:
    new_V = []
    for s in states:
        cost0 = cost(s, 0) + GAMMA * preCost(remaining_people(s, 0), V)
        cost1 = cost(s, 1) + GAMMA * preCost(remaining_people(s, 1), V)
        if cost0 > cost1:
            new_V.append([1, cost1])
        else:
            new_V.append([0, cost0])

    delta = max(abs(new_V[s][1] - V[s][1]) for s in states)
    V = new_V
    iterations += 1

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(states, [V[s][1] for s in states], color="#2563eb", linewidth=2)
    ax.set(
        title="Shuttle dispatch: optimal value (infinite horizon)",
        xlabel="Number of waiting passengers",
        ylabel="Minimum expected discounted total cost",
        xlim=(0, MAX_PEOPLE),
    )
    ax.text(0.03, 0.95, f"Discount factor = {GAMMA}", transform=ax.transAxes, va="top")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    output_path = Path(__file__).resolve().parent / "value_iteration_value.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Iterations: {iterations}; maximum change: {delta:.3e}; tolerance: {TOL}")
    print(f"Saved plot: {output_path}")
