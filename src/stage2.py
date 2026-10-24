"""Stage 2: propagate stage-1 scores across the economic graph (label-free at test time).

A planted scenario spans several payments; the quiet ones sit next to the loud
ones. For each edge we aggregate stage-1 scores of *other* edges that share the
sender, the receiver, the pair, the downstream wallet (receiver's later sends),
the upstream wallet (sender's earlier receipts) and the Louvain community.

Honest evaluation: for every outer fold, training rows get inner out-of-fold
stage-1 scores and may only look at training-row neighbours; validation rows
get scores from a model that never saw their labels and may only look at
validation-row neighbours (exactly the situation of an independent test graph).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, N_FOLDS, SEED, SUBMISSION
from src.graph_build import build_graph, static_views
from src.models.lgbm import FAST, fit_final, oof_lgbm
from src.normalize import load_split
from src.pipeline import _groups_and_folds, _labels_for, build_features_for_split, feature_matrix, log
from src.predict import verify_file, write_predictions
from src.validation import evaluate, make_folds

TIME_COLS = ("tx_hour_of_day", "tx_day_of_week", "tx_is_night", "tx_day_index", "tx_minute_of_hour")
WINDOW = 24 * 3600


def neighbour_features(g, comm: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Aggregate ``scores`` of neighbouring edges. Only edges with ``mask`` True are visible; self excluded."""
    n = g.n_edges
    s = np.where(mask, scores, np.nan)
    out = {}

    def excl_stats(keys: np.ndarray, tag: str):
        df = pd.DataFrame({"k": keys, "s": s})
        grp = df.groupby("k")["s"]
        tot, cnt = grp.transform("sum"), grp.transform("count")
        own = np.where(np.isnan(s), 0.0, s)
        own_c = (~np.isnan(s)).astype(float)
        n_other = (cnt - own_c).to_numpy()
        out[f"nb_{tag}_mean"] = np.where(n_other > 0, (tot.to_numpy() - own) / np.maximum(n_other, 1), np.nan)
        out[f"nb_{tag}_n"] = n_other
        # max of others via top-2
        mx = grp.transform("max").to_numpy()
        second = grp.transform(lambda x: x.nlargest(2).iloc[-1] if x.notna().sum() > 1 else np.nan).to_numpy()
        out[f"nb_{tag}_max"] = np.where(np.isnan(s) | (s < mx), mx, second)

    excl_stats(g.src, "snd")
    excl_stats(g.dst, "rcv")
    excl_stats(g.src * g.n_wallets + g.dst, "pair")
    excl_stats(comm[g.src], "comm")

    # time-windowed, directional neighbours through the sorted indexes
    oi, ii = g.out_index, g.in_index
    specs = {
        "down": (oi, g.dst, 0, WINDOW),      # receiver's sends after t
        "up": (ii, g.src, WINDOW, 0),        # sender's receipts before t
        "snd24": (oi, g.src, WINDOW, WINDOW),
        "rcv24": (ii, g.dst, WINDOW, WINDOW),
    }
    for tag, (index, wallet, back, fwd) in specs.items():
        lo, hi = index.bounds(wallet, g.ts, back, fwd)
        mean = np.full(n, np.nan); mx = np.full(n, np.nan); cnt = np.zeros(n)
        for e in range(n):
            a, b = int(lo[e]), int(hi[e])
            if b <= a:
                continue
            _t, _u, _c, eids = index.slice_arrays(a, b)
            vals = s[eids]
            vals = vals[(eids != e) & ~np.isnan(vals)]
            if len(vals):
                mean[e] = vals.mean(); mx[e] = vals.max(); cnt[e] = len(vals)
        out[f"nb_{tag}_mean"] = mean; out[f"nb_{tag}_max"] = mx; out[f"nb_{tag}_n"] = cnt
    f = pd.DataFrame(out)
    f["nb_chain"] = f["nb_down_max"].fillna(0) * f["nb_up_max"].fillna(0)
    f["nb_any_max"] = f[[c for c in f.columns if c.endswith("_max")]].max(axis=1)
    return f.astype("float32")


def _graph_for(name: str):
    edges = pd.read_parquet(ARTIFACTS / f"{name}_edges.parquet")
    g = build_graph(edges)
    comm = static_views(g)["community"]
    return g, comm


def _align(g, feats: pd.DataFrame) -> np.ndarray:
    """Row index into feats for each edge (edges and canonical share ledger order, but be explicit)."""
    pos = pd.Series(np.arange(len(feats)), index=feats["transaction_hash"].to_numpy())
    return pos.reindex(g.edges["transaction_hash"].to_numpy()).to_numpy()


def cv_stage2(protocol: str = "scenario", drop_time: bool = True) -> dict:
    train_dir = None
    from src.config import TRAIN_CANDIDATES, first_existing
    train_dir = first_existing(TRAIN_CANDIDATES)
    feats = build_features_for_split("train", train_dir)
    lab = _labels_for(feats, load_split(train_dir)["labels"])
    y = lab["is_suspicious"].astype(int).to_numpy()
    hashes = feats["transaction_hash"].to_numpy()
    difficulty = lab["difficulty"].to_numpy()
    X = feature_matrix(feats)
    if drop_time:
        X = X.drop(columns=[c for c in TIME_COLS if c in X.columns])
    if protocol == "scenario":
        sc = lab["scenario_id"].fillna("").astype(str).to_numpy()
        sc = np.where(sc == "", np.array([f"solo_{i}" for i in range(len(sc))]), sc)
        _, groups = np.unique(sc, return_inverse=True)
        folds = make_folds(groups, y, N_FOLDS, SEED)
    else:
        folds, _info, groups = _groups_and_folds(train_dir, feats, y)
    g, comm = _graph_for("train")
    order = _align(g, feats)
    assert not np.isnan(order.astype(float)).any()
    t0 = time.time()
    oof1 = np.full(len(y), np.nan)
    oof2 = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(folds, 1):
        # inner OOF stage-1 scores for training rows
        inner = make_folds(groups[tr], y[tr], N_FOLDS, SEED + k)
        r_in = oof_lgbm(X.iloc[tr], y[tr], inner, scale_pos_weight=3.0, **FAST)
        s_train = r_in["oof"]
        m1 = fit_final(X.iloc[tr], y[tr], int(np.median(r_in["best_iterations"])), 3.0) if False else None
        r_out = oof_lgbm(X, y, [(tr, va)], scale_pos_weight=3.0, allow_partial=True, **FAST)
        s_val = r_out["oof"][va]
        oof1[va] = s_val
        # scores per edge, masks per role
        s_all = np.full(len(y), np.nan); s_all[tr] = s_train; s_all[va] = s_val
        mask_tr = np.zeros(len(y), bool); mask_tr[tr] = True
        mask_va = np.zeros(len(y), bool); mask_va[va] = True
        nb_tr = neighbour_features(g, comm, s_all[order], mask_tr[order])
        nb_va = neighbour_features(g, comm, s_all[order], mask_va[order])
        nb = pd.DataFrame(np.nan, index=range(len(y)), columns=nb_tr.columns, dtype="float32")
        nb.iloc[order[mask_tr[order]]] = nb_tr[mask_tr[order]].to_numpy()
        nb.iloc[order[mask_va[order]]] = nb_va[mask_va[order]].to_numpy()
        X2 = pd.concat([X.reset_index(drop=True), nb], axis=1)
        r2 = oof_lgbm(X2, y, [(tr, va)], scale_pos_weight=3.0, allow_partial=True, **FAST)
        oof2[va] = r2["oof"][va]
        log(f"[{protocol}] fold {k}: stage1 AP={100 * evaluate(y[va], oof1[va], hashes[va])['ap']:.2f} "
            f"stage2 AP={100 * evaluate(y[va], oof2[va], hashes[va])['ap']:.2f} ({time.time() - t0:.0f}s)")
    e1 = evaluate(y, oof1, hashes, difficulty)
    e2 = evaluate(y, oof2, hashes, difficulty)
    rep = {"protocol": protocol, "drop_time": drop_time, "stage1": e1, "stage2": e2,
           "stage1_fold_ap": [evaluate(y[va], oof1[va], hashes[va])["ap"] for _, va in folds],
           "stage2_fold_ap": [evaluate(y[va], oof2[va], hashes[va])["ap"] for _, va in folds]}
    log(f"[{protocol}] pooled stage1 AP={100 * e1['ap']:.2f} stage2 AP={100 * e2['ap']:.2f}; "
        f"hard recall@1% {e1.get('recall_at_1pct_hard', 0):.3f} -> {e2.get('recall_at_1pct_hard', 0):.3f}; "
        f"medium {e1.get('recall_at_1pct_medium', 0):.3f} -> {e2.get('recall_at_1pct_medium', 0):.3f}")
    (ARTIFACTS / f"stage2_cv_{protocol}.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    return rep


def final_stage2(drop_time: bool = True, n_estimators: int | None = None, output: Path | None = None,
                 stage2_trees: int | None = None) -> dict:
    from src.config import TEST_CANDIDATES, TRAIN_CANDIDATES, first_existing
    train_dir, test_dir = first_existing(TRAIN_CANDIDATES), first_existing(TEST_CANDIDATES)
    feats = build_features_for_split("train", train_dir)
    lab = _labels_for(feats, load_split(train_dir)["labels"])
    y = lab["is_suspicious"].astype(int).to_numpy()
    X = feature_matrix(feats)
    if drop_time:
        X = X.drop(columns=[c for c in TIME_COLS if c in X.columns])
    feats_te = build_features_for_split("test", test_dir)
    Xte = feats_te.reindex(columns=X.columns).astype("float32")
    val = json.loads((ARTIFACTS / "validation.json").read_text())
    n_est = int(n_estimators or val.get("chosen_n_estimators", 1347))
    # stage 1: inner OOF for train (neighbour features), full model for test
    sc = lab["scenario_id"].fillna("").astype(str).to_numpy()
    sc = np.where(sc == "", np.array([f"solo_{i}" for i in range(len(sc))]), sc)
    _, groups = np.unique(sc, return_inverse=True)
    folds = make_folds(groups, y, N_FOLDS, SEED)
    r1 = oof_lgbm(X, y, folds, scale_pos_weight=3.0)
    s_train = r1["oof"]
    log(f"final stage-1 scenario-grouped OOF AP={100 * r1['mean_ap']:.2f} (full settings)")
    m1 = fit_final(X, y, n_est, 3.0)
    s_test = m1.predict_proba(Xte)[:, 1]
    g_tr, comm_tr = _graph_for("train")
    g_te, comm_te = _graph_for("test")
    o_tr, o_te = _align(g_tr, feats), _align(g_te, feats_te)
    nb_tr = neighbour_features(g_tr, comm_tr, s_train[o_tr], np.ones(len(o_tr), bool))
    nb_te = neighbour_features(g_te, comm_te, s_test[o_te], np.ones(len(o_te), bool))
    NB_tr = pd.DataFrame(np.nan, index=range(len(y)), columns=nb_tr.columns, dtype="float32"); NB_tr.iloc[o_tr] = nb_tr.to_numpy()
    NB_te = pd.DataFrame(np.nan, index=range(len(feats_te)), columns=nb_te.columns, dtype="float32"); NB_te.iloc[o_te] = nb_te.to_numpy()
    X2 = pd.concat([X.reset_index(drop=True), NB_tr], axis=1)
    X2te = pd.concat([Xte.reset_index(drop=True), NB_te], axis=1)
    if stage2_trees is None:
        r2 = oof_lgbm(X2, y, folds, scale_pos_weight=3.0)
        log(f"final stage-2 scenario-grouped OOF AP={100 * r2['mean_ap']:.2f} (full settings), best iters {r2['best_iterations']}")
        stage2_trees = int(np.median(r2["best_iterations"]))
    else:
        r2 = {"mean_ap": None, "fold_ap": None, "best_iterations": [stage2_trees]}
        log(f"final stage-2 with fixed {stage2_trees} trees")
    m2 = fit_final(X2, y, stage2_trees, 3.0)
    s2 = m2.predict_proba(X2te)[:, 1]
    output = output or (SUBMISSION / "predictions_stage2.csv")
    path = write_predictions(feats_te["transaction_hash"], s2, output)
    tx = pd.read_parquet(test_dir / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
    v = verify_file(path, tx)
    joblib.dump({"stage1": m1, "stage2": m2, "columns": list(X.columns), "nb_columns": list(nb_tr.columns)}, ARTIFACTS / "lgbm_stage2.joblib")
    pd.DataFrame({"TRANSACTION_HASH": feats_te["transaction_hash"], "stage1": s_test, "stage2": s2}).to_parquet(ARTIFACTS / "test_scores_stage2.parquet", index=False)
    rep = {"stage1_oof_scenario": r1["mean_ap"], "stage1_folds": r1["fold_ap"], "stage2_oof_scenario": r2["mean_ap"],
           "stage2_folds": r2["fold_ap"], "stage2_best_iterations": r2["best_iterations"], "verify": v}
    (ARTIFACTS / "stage2_final.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    log(f"wrote {path}: {v}")
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        cv_stage2(sys.argv[2] if len(sys.argv) > 2 else "scenario")
    elif cmd == "final":
        trees = int(sys.argv[2]) if len(sys.argv) > 2 else None
        out = SUBMISSION / (f"predictions_stage2_{trees}.csv" if trees else "predictions_stage2.csv")
        final_stage2(stage2_trees=trees, output=out)
