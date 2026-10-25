"""Small real-environment runs: four variants and exact checkpoint-resume equivalence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parent


def run(folder, game, baseline, episodes, evaluations, resume=False):
    command = [sys.executable, str(ROOT / "train.py"), "--game", game, "--baseline", baseline,
               "--episodes", str(episodes), "--eval-episodes", str(evaluations),
               "--output", str(folder)]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="mlds-pg-check-") as tmp:
        folder = Path(tmp)
        for game in ("cartpole", "pong"):
            for baseline in ("none", "ema"):
                target = folder / f"{game}_{baseline}"
                run(target, game, baseline, 1, 2)
                metrics = json.loads((target / "metrics.json").read_text())
                assert metrics["training_episodes"] == 1 and metrics["test_episodes"] == 2
                assert json.loads((target / "status.json").read_text())["status"] == "complete"
                print("Real-environment smoke passed:", game, baseline, flush=True)

        resumed, full = folder / "resumed", folder / "full"
        run(resumed, "cartpole", "ema", 2, 2)
        # 故意污染旧模型测试分数，续训后必须根据新模型指纹重新测试。
        saved_path = resumed / "evaluation_scores.json"
        saved = json.loads(saved_path.read_text())
        saved["scores"] = [999, 999]
        saved_path.write_text(json.dumps(saved))
        run(resumed, "cartpole", "ema", 3, 3, resume=True)
        run(full, "cartpole", "ema", 3, 3)
        a = torch.load(resumed / "checkpoint.pt", weights_only=True)
        b = torch.load(full / "checkpoint.pt", weights_only=True)
        assert a["history"] == b["history"] and a["baseline_value"] == b["baseline_value"]
        assert all(torch.equal(a["policy"][key], b["policy"][key]) for key in a["policy"])
        scores_a = json.loads((resumed / "metrics.json").read_text())["test_scores"]
        scores_b = json.loads((full / "metrics.json").read_text())["test_scores"]
        assert scores_a == scores_b and 999 not in scores_a
        # 扩大测试局数时，只补做新局数。
        run(resumed, "cartpole", "ema", 3, 4, resume=True)
        expanded = json.loads((resumed / "metrics.json").read_text())["test_scores"]
        assert len(expanded) == 4 and expanded[:3] == scores_a
        result = {"four_environment_variants_passed": True, "resume_matches_uninterrupted_training": True,
                  "stale_evaluation_rejected": True, "evaluation_resume_passed": True}
        (ROOT / "resume_validation.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
