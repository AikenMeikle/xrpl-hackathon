"""Build the code ZIP for the organiser portal (no data, no secrets, <= 8 MiB)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.config import ROOT

INCLUDE_FILES = ["run.py", "payments.py", "requirements.txt", "README.md", "LICENSE", "prompts.txt"]
INCLUDE_DIRS = ["src", "tests", "docs", "notebooks"]
INCLUDE_ARTIFACTS = ["validation.json", "manifest.json", "inspection_report.md", "feature_dictionary.csv",
                     "lgbm_gain_importance.csv", "baseline_metrics.json", "test_alert_evidence.csv", "train_case_evidence.json"]
EXCLUDE_SUFFIX = (".pyc", ".parquet", ".zip", ".pt", ".joblib")


def package(out: Path | None = None) -> Path:
    out = out or (ROOT / "submission" / "code_submission.zip")
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            p = ROOT / f
            if p.exists():
                z.write(p, f)
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(EXCLUDE_SUFFIX) and ".ipynb_checkpoints" not in p.parts:
                    z.write(p, str(p.relative_to(ROOT)))
        for f in INCLUDE_ARTIFACTS:
            p = ROOT / "artifacts" / f
            if p.exists():
                z.write(p, f"artifacts/{f}")
    size = out.stat().st_size
    if size > 8 * 1024 * 1024:
        raise AssertionError(f"code zip is {size} bytes > 8 MiB")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        if len(names) > 2000:
            raise AssertionError("too many files in zip")
        if any(n.endswith(".parquet") or n.startswith("data/") for n in names):
            raise AssertionError("data leaked into zip")
    return out
