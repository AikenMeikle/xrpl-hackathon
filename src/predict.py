"""Submission writer, verifier and rank blending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import EXPECTED_TEST_ROWS


def rank_percentile(scores: np.ndarray) -> np.ndarray:
    """Average-rank percentile strictly inside (0, 1)."""
    r = rankdata(scores, method="average")
    return (r - 0.5) / len(r)


def blend(lgbm_scores: np.ndarray, gnn_scores: np.ndarray | None, gnn_weight: float) -> np.ndarray:
    if gnn_scores is None or gnn_weight <= 0:
        return rank_percentile(lgbm_scores)
    return rank_percentile((1 - gnn_weight) * rank_percentile(lgbm_scores) + gnn_weight * rank_percentile(gnn_scores))


def write_predictions(hashes: pd.Series, scores: np.ndarray, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"TRANSACTION_HASH": hashes.astype(str).to_numpy(), "score": np.asarray(scores, dtype=float)})
    df.to_csv(output, index=False, float_format="%.12g", lineterminator="\n")
    return output


def verify_file(path: Path, expected_hashes: pd.Series | None = None, expected_rows: int = EXPECTED_TEST_ROWS) -> dict:
    path = Path(path)
    problems: list[str] = []
    size = path.stat().st_size
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
    if header.split(",")[:2] != ["TRANSACTION_HASH", "score"]:
        problems.append(f"header is {header!r}, expected TRANSACTION_HASH,score")
    df = pd.read_csv(path, dtype={"TRANSACTION_HASH": str})
    if len(df) != expected_rows:
        problems.append(f"{len(df)} rows, expected {expected_rows}")
    if df["TRANSACTION_HASH"].duplicated().any():
        problems.append("duplicate hashes")
    s = pd.to_numeric(df["score"], errors="coerce")
    if s.isna().any() or not np.isfinite(s).all():
        problems.append("non-finite scores")
    if ((s < 0) | (s > 1)).any():
        problems.append("scores outside [0,1]")
    if s.nunique() < 2:
        problems.append("scores are constant")
    if s.isin([0.0, 1.0]).all():
        problems.append("scores are 0/1 only")
    if size > 12 * 1024 * 1024:
        problems.append(f"file is {size} bytes > 12 MiB")
    if expected_hashes is not None:
        exp = set(expected_hashes.astype(str))
        got = set(df["TRANSACTION_HASH"])
        if exp != got:
            problems.append(f"hash set mismatch: missing={len(exp - got)} extra={len(got - exp)}")
    return {
        "file": str(path), "rows": int(len(df)), "bytes": int(size), "unique_scores": int(s.nunique()),
        "min": float(s.min()), "max": float(s.max()), "tie_fraction": float(1 - s.nunique() / max(len(s), 1)),
        "ok": not problems, "problems": problems,
    }
