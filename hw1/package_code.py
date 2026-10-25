"""Create a small upload bundle containing code/docs, not local environments or trials."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {".venv", "__pycache__", ".git", "results", "smoke", "tmp", "output", "tuning_results"}

if __name__ == "__main__":
    target = ROOT / "hw1_code.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
                continue
            if path.name == "report_results.md":
                continue
            if path.suffix in {".py", ".sh", ".md"} or path.name in {"requirements.txt", ".gitignore"}:
                archive.write(path, Path("hw1") / relative)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(".venv/" in name or "/results/" in name for name in archive.namelist())
        print(f"Packaged {len(archive.namelist())} code/documentation files: {target}")
