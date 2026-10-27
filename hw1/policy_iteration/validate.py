"""Analytic and cross-method checks for shuttle dispatch."""

import argparse
import json
import runpy
from pathlib import Path

import main as model


def validate_zero_discount():
    original_gamma = model.GAMMA
    model.GAMMA = 0.0

    try:

        initial_policy = [1] * (model.MAX_PEOPLE + 1)
        values, _, _ = model.evaluate_policy(initial_policy)
        improved_policy = model.improve_policy(initial_policy, values)
        improved_values, _, _ = model.evaluate_policy(improved_policy)


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


    # V(200)=400/(1-gamma)；V(199)=398+gamma*V(200)。

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
    hw1_folder = Path(__file__).resolve().parent.parent
    finite = runpy.run_path(str(hw1_folder / "main.py"))
    vi = runpy.run_path(str(hw1_folder / "value_iteration.py"))
    pi_policy, pi_values, history = model.policy_iteration()



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
            print(f"Zero discount: correct actions {result['correct_actions']}/{result['total_states']}")
        elif name == "boundaries":
            print(f"Boundary cases: {len(result['one_step_checks'])} cost checks passed")
            print(f"Cap transitions: {len(result['transition_checks'])} checks passed")
            for row in result["long_run_checks"]:
                print(
                    f"Always wait, initial state {row['state']}: "
                    f"expected cost {row['expected_value']:.8f}, "
                    f"computed cost {row['actual_value']:.8f}"
                )
        else:
            for row in result["comparisons"]:
                print(
                    f"{row['algorithm_a']} vs {row['algorithm_b']}: "
                    f"max value difference {row['max_cost_difference']:.9g}, "
                    f"matching actions {row['matching_actions']}/{result['states_checked']}"
                )
        if name != "comparison":
            print(f"Max analytic error: {result['max_value_error']:.3e}")
        print(f"Passed; saved results: {output_path}")
