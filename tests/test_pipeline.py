"""Contract tests for the model / submission layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FORBIDDEN_FEATURE_PIECES
from src.models.lgbm import assert_no_forbidden, fit_final
from src.predict import blend, rank_percentile, verify_file, write_predictions
from src.validation import evaluate, make_folds


def test_forbidden_identifier_columns_are_rejected():
    for bad in ("transaction_hash", "source_wallet", "dst_address", "scenario_id", "cohort_id", "family", "difficulty"):
        with pytest.raises(AssertionError):
            assert_no_forbidden(["tx_log_paid_usd", bad])
    assert_no_forbidden(["tx_log_paid_usd", "fanin_sweep_quality"])


def test_labels_never_enter_feature_tables():
    import inspect
    import src.features_motifs, src.features_transaction, src.features_wallet, src.graph_build, src.normalize
    for mod in (src.features_motifs, src.features_transaction, src.features_wallet, src.graph_build):
        source = inspect.getsource(mod)
        assert "is_suspicious" not in source and "labels" not in source and "scenario_id" not in source
    # normalize() itself (not the loader) never touches labels
    source = inspect.getsource(src.normalize.normalize)
    assert "label" not in source and "is_suspicious" not in source


def test_model_scores_are_finite_in_unit_interval_and_submission_verifies(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"tx_f{i}" for i in range(6)])
    y = (X["tx_f0"] + rng.normal(scale=0.5, size=n) > 1.5).astype(int).to_numpy()
    model = fit_final(X, y, n_estimators=50)
    s = model.predict_proba(X)[:, 1]
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    hashes = pd.Series([f"{i:064X}" for i in range(n)])
    p = write_predictions(hashes, s, tmp_path / "pred.csv")
    v = verify_file(p, hashes, expected_rows=n)
    assert v["ok"], v["problems"]
    # duplicates / missing / out of range are caught
    bad = pd.read_csv(p); bad.loc[0, "score"] = 1.5; bad.to_csv(tmp_path / "bad.csv", index=False)
    assert not verify_file(tmp_path / "bad.csv", hashes, expected_rows=n)["ok"]
    dup = pd.concat([pd.read_csv(p), pd.read_csv(p).head(1)]); dup.to_csv(tmp_path / "dup.csv", index=False)
    assert not verify_file(tmp_path / "dup.csv", hashes, expected_rows=n)["ok"]


def test_rank_blend_strictly_inside_unit_interval_without_rounding_ties():
    a = np.random.default_rng(1).normal(size=5000)
    b = np.random.default_rng(2).normal(size=5000)
    r = rank_percentile(a)
    assert r.min() > 0 and r.max() < 1 and len(np.unique(r)) == 5000
    assert np.array_equal(np.argsort(r), np.argsort(a))
    z = blend(a, b, 0.3)
    assert z.min() > 0 and z.max() < 1
    assert np.array_equal(np.argsort(blend(a, None, 0.0)), np.argsort(a))


def test_group_folds_keep_groups_and_hashes_apart():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, size=3000)
    y = (rng.random(3000) < 0.05).astype(int)
    folds = make_folds(groups, y, 3)
    for tr, va in folds:
        assert not (set(tr) & set(va))
        assert not (set(groups[tr]) & set(groups[va]))
    covered = np.concatenate([va for _, va in folds])
    assert len(covered) == 3000 and len(np.unique(covered)) == 3000


def test_evaluate_reports_ap_and_top1pct():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 20)
    s = np.linspace(1, 0, 200)
    h = np.array([f"{i:04d}" for i in range(200)])
    r = evaluate(y, s, h)
    assert 0 <= r["ap"] <= 1 and "precision_at_1pct" in r and r["n"] == 200
