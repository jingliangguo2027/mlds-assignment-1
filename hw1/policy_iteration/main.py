"""班车调度：先评估固定策略，再改进策略，直到动作不再改变。"""

MAX_PEOPLE = 200
K = 15
GAMMA = 0.95
TOL = 1e-8

states = list(range(MAX_PEOPLE + 1))

def remaining_people(s, a):
    if a == 0:
        return s
    return max(s - K, 0)

def cost(s, a):
    return 100 * a + 2 * remaining_people(s, a)

def preCost(r, values):
    """剩余 r 人之后，随机到达 1～5 人，对下一状态的价值取平均。"""
    totalCost = 0.0
    for k in range(1, 6):
        next_s = min(MAX_PEOPLE, r + k)
        totalCost += values[next_s]
    return totalCost / 5

def evaluate_policy(policy):
    """第一步：固定 policy，反复更新到价值稳定；这里不选择动作。"""
    values = [0.0] * (MAX_PEOPLE + 1)
    evaluation_steps = 0

    while True:
        new_values = [0.0] * (MAX_PEOPLE + 1)

        for s in states:
            a = policy[s]  # 动作已规定，直接服从，不比较 0 和 1。
            r = remaining_people(s, a)
            new_values[s] = cost(s, a) + GAMMA * preCost(r, values)

        delta = max(abs(new_values[s] - values[s]) for s in states)
        values = new_values
        evaluation_steps += 1

        if delta < TOL:
            return values, evaluation_steps, delta

def improve_policy(policy, values):
    """第二步：查已经评估好的 values，为每种人数重新比较两个动作。"""
    new_policy = policy.copy()

    for s in states:
        cost0 = cost(s, 0) + GAMMA * preCost(remaining_people(s, 0), values)
        cost1 = cost(s, 1) + GAMMA * preCost(remaining_people(s, 1), values)

        if cost1 < cost0:
            new_policy[s] = 1
        elif cost0 < cost1:
            new_policy[s] = 0

    return new_policy

def policy_iteration():
    policy = [0] * (MAX_PEOPLE + 1)
    history = []

    while True:
        values, evaluation_steps, delta = evaluate_policy(policy)

        new_policy = improve_policy(policy, values)
        changed_states = sum(new_policy[s] != policy[s] for s in states)
        iteration = len(history) + 1

        history.append({
            "iteration": iteration,
            "evaluation_steps": evaluation_steps,
            "evaluation_delta": delta,
            "changed_states": changed_states,
            "old_action_at_20": policy[20],
            "new_action_at_20": new_policy[20],
            "old_policy_value_at_20": values[20],
        })
        print(
            f"第 {iteration} 轮：评估更新 {evaluation_steps} 次；"
            f"旧策略在 20 人时成本 {values[20]:.2f}；"
            f"20 人时动作 {policy[20]} -> {new_policy[20]}；"
            f"共修改 {changed_states} 个状态"
        )

        if changed_states == 0:
            return policy, values, history

        policy = new_policy

if __name__ == "__main__":
    import json
    from pathlib import Path
    from plot_policy import plot_policy

    policy, values, history = policy_iteration()
    output_folder = Path(__file__).resolve().parent
    results = {
        "parameters": {"max_people": MAX_PEOPLE, "capacity": K, "gamma": GAMMA, "tolerance": TOL},
        "policy": policy,
        "values": values,
        "history": history,
    }
    results_path = output_folder / "results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    image_path = plot_policy(states, policy, GAMMA, output_folder)
    print(f"结果已保存：{results_path}")
    print(f"策略图已保存：{image_path}")
