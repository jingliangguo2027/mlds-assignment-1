MAX_PEOPLE = 200

states = list(range(MAX_PEOPLE + 1))

actions = [0, 1]
K = 15 

T=500

V = [
    [[0, 0.0] for _ in range(MAX_PEOPLE + 1)] for _ in range(T + 1)
]
def remaining_people(s, a):
    if a == 0:
        return s
    if a == 1:
        return max(s - K,0)

def cost(s,a):
    return 100 * a + remaining_people(s,a) * 2

def preCost(s,t):
    totalCost = 0
    for k in range(1,6):
        next_s = min(MAX_PEOPLE, s + k)
        totalCost += V[t+1][next_s][1]
    return totalCost / 5

for t in range(T - 1, -1, -1):
    for s in range(MAX_PEOPLE + 1):
        cost0 = cost(s,0) + 0.95 * preCost(s,t)
        cost1 = cost(s,1) + 0.95 * preCost(max(s-15,0),t)
        if cost0 > cost1:
            V[t][s] = [1,cost1]
        else:
            V[t][s] = [0,cost0]

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial_costs = [V[0][s][1] for s in states]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(states, initial_costs, color="#2563eb", linewidth=2)
    ax.set(
        title=f"Shuttle dispatch: optimal value at time 0 (T = {T})",
        xlabel="Initial number of waiting passengers",
        ylabel="Minimum expected discounted total cost",
        xlim=(0, MAX_PEOPLE),
    )
    ax.text(0.03, 0.95, "Discount factor = 0.95", transform=ax.transAxes, va="top")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    output_path = Path(__file__).resolve().parent / "finite_horizon_value.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_path}")

