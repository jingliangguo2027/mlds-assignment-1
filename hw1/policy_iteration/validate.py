"""班车验证实验：零折扣、人数边界、三种算法结果对照。"""

import argparse
import json
import runpy
from pathlib import Path

import main as model


def validate_zero_discount():
    original_gamma = model.GAMMA
    model.GAMMA = 0.0

    try:
        # 故意从错误策略开始：所有人数都发车。
        initial_policy = [1] * (model.MAX_PEOPLE + 1)
        values, _, _ = model.evaluate_policy(initial_policy)
        improved_policy = model.improve_policy(initial_policy, values)
        improved_values, _, _ = model.evaluate_policy(improved_policy)

        # gamma=0 时，理论最优成本为 2*s，动作全部为 0。
        correct_actions = sum(improved_policy[s] == 0 for s in model.states)
        max_value_error = max(
            abs(improved_values[s] - 2 * s) for s in model.states
        )
        assert correct_actions == len(model.states)
        assert max_value_error == 0.0

        return {
            "experiment": "zero_discount",
            "gamma": 0.0,
            "initial_policy": "dispatch at every state",
            "expected_policy": "wait at every state",
            "correct_actions": correct_actions,
            "total_states": len(model.states),
            "max_value_error": max_value_error,
            "passed": True,
        }
    finally:
        model.GAMMA = original_gamma


def validate_boundaries():
    # 手算发车后的剩余人数和本轮成本，覆盖空站、不满一车、刚好一车和满站。
    cases = [
        (0, 0, 0, 0),
        (0, 1, 0, 100),
        (8, 1, 0, 100),
        (15, 1, 0, 100),
        (16, 1, 1, 102),
        (200, 0, 200, 400),
        (200, 1, 185, 470),
    ]
    one_step_checks = []
    for s, action, expected_remaining, expected_cost in cases:
        actual_remaining = model.remaining_people(s, action)
        actual_cost = model.cost(s, action)
        assert actual_remaining == expected_remaining
        assert actual_cost == expected_cost
        one_step_checks.append({
            "state": s,
            "action": action,
            "remaining": actual_remaining,
            "cost": actual_cost,
        })

    # 用非线性的已知价值检查上限处的概率合并，而不只检查下标是否越界。
    # 例如剩 198 人时，下一状态为 199、200、200、200、200。
    probe_values = [s * s for s in model.states]
    transition_cases = [
        (198, [199, 200, 200, 200, 200]),
        (199, [200, 200, 200, 200, 200]),
        (200, [200, 200, 200, 200, 200]),
    ]
    transition_checks = []
    for remaining, expected_states in transition_cases:
        expected = sum(s * s for s in expected_states) / 5
        actual = model.preCost(remaining, probe_values)
        assert abs(actual - expected) < 1e-9
        transition_checks.append({
            "remaining": remaining,
            "expected_next_states": expected_states,
            "expected_future_value": expected,
            "actual_future_value": actual,
        })

    # 再验证真实的长期成本。固定永不发车策略，满站后每轮均有 200 人。
    # V(200)=400/(1-gamma)；V(199)=398+gamma*V(200)。
    # 从 198 人出发，下轮以 1/5 概率到 199，以 4/5 概率到 200。
    gamma = model.GAMMA
    exact_200 = 400 / (1 - gamma)
    exact_199 = 398 + gamma * exact_200
    exact_198 = 396 + gamma * (exact_199 / 5 + 4 * exact_200 / 5)
    expected_values = {198: exact_198, 199: exact_199, 200: exact_200}
    values, evaluation_steps, _ = model.evaluate_policy([0] * len(model.states))
    allowed_error = model.TOL / (1 - gamma)
    long_run_checks = []
    for s, expected in expected_values.items():
        error = abs(values[s] - expected)
        assert error <= allowed_error + 1e-9
        long_run_checks.append({
            "state": s,
            "expected_value": expected,
            "actual_value": values[s],
            "absolute_error": error,
        })

    return {
        "experiment": "boundaries",
        "gamma": gamma,
        "queue_cap_assumption": "arrivals above 200 are excluded from the queue",
        "one_step_checks": one_step_checks,
        "transition_checks": transition_checks,
        "long_run_checks": long_run_checks,
        "evaluation_steps": evaluation_steps,
        "max_value_error": max(row["absolute_error"] for row in long_run_checks),
        "allowed_value_error": allowed_error,
        "passed": True,
    }


def validate_algorithm_comparison():
    # 重新运行三个算法，不读取以前保存的结果。
    # run_path 默认不会执行文件中 __name__ == "__main__" 的画图部分。
    hw1_folder = Path(__file__).resolve().parent.parent
    finite = runpy.run_path(str(hw1_folder / "main.py"))
    vi = runpy.run_path(str(hw1_folder / "value_iteration.py"))
    pi_policy, pi_values, history = model.policy_iteration()

    # 统一成两个数组：(所有人数的成本, 所有人数的动作)。
    # 有限期只取第 0 轮，与无限期从相同人数出发的结果比较。
    results = {
        "finite_500": (
            [finite["V"][0][s][1] for s in model.states],
            [finite["V"][0][s][0] for s in model.states],
        ),
        "value_iteration": (
            [vi["V"][s][1] for s in model.states],
            [vi["V"][s][0] for s in model.states],
        ),
        "policy_iteration": (pi_values, pi_policy),
    }

    # 两种无限期方法的停止精度不同，500 轮也有很小的截断误差。
    # 因此成本允许 0.0001 的差异；本题这组参数下，动作预计完全一致。
    tolerance = 1e-4
    comparisons = []
    pairs = [
        ("finite_500", "value_iteration"),
        ("finite_500", "policy_iteration"),
        ("value_iteration", "policy_iteration"),
    ]
    for name_a, name_b in pairs:
        values_a, policy_a = results[name_a]
        values_b, policy_b = results[name_b]
        max_difference = max(abs(values_a[s] - values_b[s]) for s in model.states)
        different_actions = [s for s in model.states if policy_a[s] != policy_b[s]]

        assert max_difference < tolerance, (name_a, name_b, max_difference)
        assert not different_actions, (name_a, name_b, different_actions)
        comparisons.append({
            "algorithm_a": name_a,
            "algorithm_b": name_b,
            "max_cost_difference": max_difference,
            "matching_actions": len(model.states) - len(different_actions),
            "different_action_states": different_actions,
        })

    return {
        "experiment": "algorithm_comparison",
        "states_checked": len(model.states),
        "cost_tolerance": tolerance,
        "finite_horizon": finite["T"],
        "value_iteration_updates": vi["iterations"],
        "policy_iteration_rounds": len(history),
        "comparisons": comparisons,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["gamma0", "boundaries", "comparison", "all"], default="all")
    args = parser.parse_args()
    experiments = {
        "gamma0": validate_zero_discount,
        "boundaries": validate_boundaries,
        "comparison": validate_algorithm_comparison,
    }
    selected = list(experiments) if args.experiment == "all" else [args.experiment]

    for name in selected:
        result = experiments[name]()
        output_path = Path(__file__).resolve().parent / f"validation_{name}.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if name == "gamma0":
            print(f"零折扣：正确动作 {result['correct_actions']}/{result['total_states']}")
        elif name == "boundaries":
            print(f"人数边界：{len(result['one_step_checks'])} 个本轮成本检查通过")
            print(f"上限处概率转移：{len(result['transition_checks'])} 个检查通过")
            for row in result["long_run_checks"]:
                print(
                    f"永不发车，初始 {row['state']} 人："
                    f"理论成本 {row['expected_value']:.8f}，"
                    f"计算成本 {row['actual_value']:.8f}"
                )
        else:
            for row in result["comparisons"]:
                print(
                    f"{row['algorithm_a']} vs {row['algorithm_b']}："
                    f"最大成本差 {row['max_cost_difference']:.9g}，"
                    f"动作一致 {row['matching_actions']}/{result['states_checked']}"
                )
        if name != "comparison":
            print(f"与理论成本的最大误差：{result['max_value_error']:.3e}")
        print(f"验证通过，结果保存到：{output_path}")
