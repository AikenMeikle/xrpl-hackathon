"""Orchestration: features -> blocked validation -> final fit -> prediction."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (ARTIFACTS, EXPECTED_TEST_ROWS, GNN_BLEND_WEIGHTS, LGBM_PARAMS, N_FOLDS, SEED, SUBMISSION)
from src.features_transaction import transaction_features
from src.features_wallet import wallet_features
from src.graph_build import build_graph, static_views
from src.models.lgbm import FAST, assert_no_forbidden, fit_final, gain_importance, oof_lgbm
from src.normalize import economic_edges, load_split, normalize
from src.predict import blend, rank_percentile, verify_file, write_predictions
from src.validation import evaluate, flow_groups, make_folds, time_holdout, wallet_overlap

_T0 = time.time()
ID_COLS = ["transaction_hash", "ledger_index", "transaction_index", "timestamp_unix", "in_economic_graph"]
FEATURE_FAMILIES = {
    "basic_transaction": ("tx_",),
    "wallet_windows": ("full_", "asof_", "win_", "fwd_"),
    "value_forwarding": ("match_",),
    "motif_fanin": ("fanin_",),
    "motif_fanout": ("fanout_",),
    "motif_scatter_gather": ("sg_",),
    "motif_gather_scatter": ("gs_",),
    "motif_cycle": ("cycle_",),
    "motif_stack": ("chain_",),
    "motif_bipartite": ("bip_",),
    "motif_irregular": ("irr_",),
}


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def build_features_for_split(name: str, directory: Path, force: bool = False) -> pd.DataFrame:
    out = ARTIFACTS / f"{name}_features.parquet"
    if out.exists() and not force:
        log(f"{name}: reusing {out}")
        return pd.read_parquet(out)
    t0 = time.time()
    split = load_split(directory)
    canonical, legs = normalize(split["transactions"], split["transfers"], split["rates"])
    edges = economic_edges(canonical)
    log(f"{name}: {len(canonical):,} tx, {len(edges):,} economic edges, {len(legs):,} legs, "
        f"{int((~canonical['success']).sum())} failed, rate_missing={int(canonical['paid_rate_missing'].sum())}")
    if canonical["paid_rate_missing"].any() or canonical["delivered_rate_missing"].any():
        log("  WARNING: some payments have no reference rate for their hour; kept as NaN + flag")
    canonical.to_parquet(ARTIFACTS / f"{name}_canonical.parquet", index=False)
    legs.to_parquet(ARTIFACTS / f"{name}_legs.parquet", index=False)
    edges.to_parquet(ARTIFACTS / f"{name}_edges.parquet", index=False)

    g = build_graph(edges)
    views = static_views(g)
    log(f"{name}: graph {g.n_wallets} wallets, components={len(np.unique(views['component']))}, "
        f"communities={len(np.unique(views['community']))}")

    tx_f = transaction_features(canonical)
    tx_f.insert(0, "transaction_hash", canonical["transaction_hash"].to_numpy())
    log(f"{name}: transaction features {tx_f.shape[1] - 1}")

    wf, prof = wallet_features(g, views)
    wf.insert(0, "transaction_hash", g.edges["transaction_hash"].to_numpy())
    log(f"{name}: wallet/window features {wf.shape[1] - 1}")
    prof.to_parquet(ARTIFACTS / f"{name}_wallet_profile.parquet")

    try:
        from src.features_motifs import motif_features
        mf, witnesses = motif_features(g, views, prof)
        mf.insert(0, "transaction_hash", g.edges["transaction_hash"].to_numpy())
        (ARTIFACTS / f"{name}_witnesses.json").write_text(json.dumps(witnesses, default=str), encoding="utf-8")
        log(f"{name}: motif features {mf.shape[1] - 1}")
    except ImportError:
        mf = pd.DataFrame({"transaction_hash": g.edges["transaction_hash"].to_numpy()})
        log(f"{name}: motif module not available yet")

    feats = tx_f.merge(wf, on="transaction_hash", how="left").merge(mf, on="transaction_hash", how="left")
    if len(feats) != len(canonical):
        raise AssertionError("feature join changed the row count")
    if feats["transaction_hash"].duplicated().any():
        raise AssertionError("duplicate hash after feature join")
    for c in ("ledger_index", "transaction_index", "timestamp_unix", "in_economic_graph"):
        feats[c] = canonical[c].to_numpy()
    feats["graph_context_missing"] = (~canonical["in_economic_graph"]).astype("float32").to_numpy()
    feats.to_parquet(out, index=False)
    log(f"{name}: wrote {out} {feats.shape} in {time.time() - t0:.1f}s")
    return feats


def feature_matrix(feats: pd.DataFrame, families: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in feats.columns if c not in ID_COLS]
    if families is not None:
        prefixes = tuple(p for fam in families for p in FEATURE_FAMILIES[fam]) + ("graph_context_missing",)
        cols = [c for c in cols if c.startswith(prefixes)]
    assert_no_forbidden(cols)
    return feats[cols].astype("float32")


def _labels_for(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.rename(columns={"TRANSACTION_HASH": "transaction_hash"}).set_index("transaction_hash")
    lab = lab.reindex(feats["transaction_hash"])
    if lab["is_suspicious"].isna().any():
        raise AssertionError("some transactions have no label")
    return lab.reset_index(drop=True)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _groups_and_folds(train_dir: Path, feats: pd.DataFrame, y: np.ndarray) -> tuple[list, dict, np.ndarray]:
    edges = pd.read_parquet(ARTIFACTS / "train_edges.parquet")
    g = build_graph(edges)
    views = static_views(g)
    groups_edge, info = flow_groups(g, community=views["community"])
    gmap = pd.Series(groups_edge, index=g.edges["transaction_hash"].to_numpy())
    groups = gmap.reindex(feats["transaction_hash"]).to_numpy()
    # non-edge transactions (none in this package) become their own singleton groups
    missing = np.isnan(groups.astype(float))
    if missing.any():
        groups = groups.astype(float)
        groups[missing] = groups[~missing].max() + 1 + np.arange(missing.sum())
    groups = groups.astype(np.int64)
    folds = make_folds(groups, y, N_FOLDS, SEED)
    info["wallet_overlap_frac"] = wallet_overlap(folds, g) if len(feats) == g.n_edges else None
    info["fold_sizes"] = [[int(len(tr)), int(len(va)), int(y[va].sum())] for tr, va in folds]
    return folds, info, groups


def _validation_inputs(train_dir: Path, force: bool = False):
    feats = build_features_for_split("train", train_dir, force=force)
    lab = _labels_for(feats, load_split(train_dir)["labels"])
    y = lab["is_suspicious"].astype(int).to_numpy()
    hashes = feats["transaction_hash"].to_numpy()
    difficulty = lab["difficulty"].to_numpy() if "difficulty" in lab else None
    X = feature_matrix(feats)
    folds, ginfo, groups = _groups_and_folds(train_dir, feats, y)
    log(f"validate: {X.shape[0]:,} rows x {X.shape[1]} features, positives={y.sum()} ({y.mean():.4f})")
    log(f"flow groups: {ginfo}")
    return feats, lab, y, hashes, difficulty, X, folds, ginfo, groups


def validate(train_dir: Path, test_dir: Path, skip_gnn: bool = False, force: bool = False, stage: str = "all") -> dict:
    """Stages can run in separate processes: lgbm | ablation | gnn | merge (or all)."""
    if stage == "merge":
        return _merge_validation(train_dir)
    feats, lab, y, hashes, difficulty, X, folds, ginfo, groups = _validation_inputs(train_dir, force)
    report: dict = {"n_rows": int(len(y)), "n_features": int(X.shape[1]), "prevalence": float(y.mean()),
                    "flow_groups": ginfo, "lgbm_params": LGBM_PARAMS, "seed": SEED}
    if stage in ("lgbm", "all"):
        _stage_lgbm(feats, lab, y, hashes, difficulty, X, folds, groups, report)
    if stage == "diagnostics":
        _stage_diagnostics(feats, lab, y, X, folds, report)
    if stage in ("ablation", "all"):
        _stage_ablation(feats, y, folds, report)
    if stage in ("gnn", "all") and not skip_gnn:
        _stage_gnn(feats, y, hashes, difficulty, folds, report)
    if stage == "all":
        return _merge_validation(train_dir)
    return report


def _stage_lgbm(feats, lab, y, hashes, difficulty, X, folds, groups, report) -> None:
    # ---- primary: LightGBM blocked OOF, full feature set -------------------
    # Positive-weight choice is made like-for-like with fast settings first.
    res_fast = oof_lgbm(X, y, folds, **FAST)
    res_w = oof_lgbm(X, y, folds, scale_pos_weight=3.0, **FAST)
    report["lgbm_fast_unweighted"] = {"mean_ap": res_fast["mean_ap"], "fold_ap": res_fast["fold_ap"]}
    report["lgbm_fast_posweight3"] = {"mean_ap": res_w["mean_ap"], "fold_ap": res_w["fold_ap"]}
    log(f"fast LightGBM: unweighted AP={100 * res_fast['mean_ap']:.2f} vs pos_weight=3 AP={100 * res_w['mean_ap']:.2f}")
    chosen_weight = 3.0 if res_w["mean_ap"] > res_fast["mean_ap"] + 0.005 else 1.0
    res = oof_lgbm(X, y, folds, scale_pos_weight=chosen_weight)
    report["lgbm_blocked"] = {**evaluate(y, res["oof"], hashes, difficulty), "fold_ap": res["fold_ap"],
                              "std_ap": res["std_ap"], "best_iterations": res["best_iterations"], "scale_pos_weight": chosen_weight}
    log(f"LightGBM blocked OOF AP={100 * res['mean_ap']:.2f} folds={[round(100 * a, 2) for a in res['fold_ap']]} "
        f"P@1%={report['lgbm_blocked']['precision_at_1pct']:.3f} R@1%={report['lgbm_blocked']['recall_at_1pct']:.3f} spw={chosen_weight}")
    imp = gain_importance(res["models"][0], X.columns)
    report["top_gain_features"] = imp.head(40).round(2).to_dict()
    report["chosen_scale_pos_weight"] = chosen_weight
    report["chosen_n_estimators"] = int(np.median(res["best_iterations"]))
    pd.DataFrame({"transaction_hash": hashes, "y": y, "group": groups, "oof_lgbm": res["oof"]}).to_parquet(ARTIFACTS / "oof_lgbm.parquet", index=False)
    (ARTIFACTS / "validation_lgbm.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    _stage_diagnostics(feats, lab, y, X, folds, report)


def _stage_diagnostics(feats, lab, y, X, folds, report) -> None:
    """Secondary diagnostics with fast LightGBM settings; each result is persisted as soon as it exists."""
    path = ARTIFACTS / "validation_diagnostics.json"
    diag: dict = {}

    def save():
        path.write_text(json.dumps(diag, indent=2, default=float), encoding="utf-8")
        report.update(diag)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    rand_folds = [(np.sort(np.concatenate([perm[j::N_FOLDS] for j in range(N_FOLDS) if j != i])), np.sort(perm[i::N_FOLDS])) for i in range(N_FOLDS)]
    tr_i, va_i = time_holdout(feats["timestamp_unix"].to_numpy())
    r_time = oof_lgbm(X, y, [(tr_i, va_i)], allow_partial=True, **FAST)
    diag["lgbm_time_holdout_last2d"] = {"ap": r_time["mean_ap"], "n_val": int(len(va_i)), "n_pos_val": int(y[va_i].sum()),
                                        "note": "features use the full window; stress test, not an as-of proxy"}
    log(f"time holdout (last 2 days, 6h embargo) AP={100 * r_time['mean_ap']:.2f}")
    save()
    asof_cols = [c for c in X.columns if c.startswith(("tx_", "asof_", "graph_context"))]
    r_asof = oof_lgbm(X[asof_cols], y, folds, **FAST)
    diag["lgbm_asof_only"] = {"mean_ap": r_asof["mean_ap"], "n_features": len(asof_cols)}
    log(f"as-of-only features blocked AP={100 * r_asof['mean_ap']:.2f} ({len(asof_cols)} features)")
    save()
    r_rand = oof_lgbm(X, y, rand_folds, **FAST)
    diag["lgbm_random_split"] = {"mean_ap": r_rand["mean_ap"], "fold_ap": r_rand["fold_ap"]}
    log(f"random-split OOF AP={100 * r_rand['mean_ap']:.2f} (optimistic upper bound)")
    save()
    if "scenario_id" in lab:
        sc = lab["scenario_id"].fillna("").astype(str).to_numpy()
        sc_groups = np.where(sc == "", np.array([f"solo_{i}" for i in range(len(sc))]), sc)
        _, sc_codes = np.unique(sc_groups, return_inverse=True)
        sc_folds = make_folds(sc_codes, y, N_FOLDS, SEED)
        r_sc = oof_lgbm(X, y, sc_folds, **FAST)
        diag["lgbm_scenario_grouped"] = {"mean_ap": r_sc["mean_ap"], "fold_ap": r_sc["fold_ap"],
                                         "note": "scenario_id used for fold assignment only"}
        log(f"scenario-grouped OOF AP={100 * r_sc['mean_ap']:.2f}")
        save()


def _stage_ablation(feats, y, folds, report) -> None:
    ledger = []
    base_ap = json.loads((ARTIFACTS / "baseline_metrics.json").read_text())["random_holdout_ap"] if (ARTIFACTS / "baseline_metrics.json").exists() else None
    ledger.append({"config": "organiser_baseline_logreg (random 20% holdout)", "ap": base_ap, "n_features": 11})
    cumulative: list[str] = []
    for fam in FEATURE_FAMILIES:
        cumulative.append(fam)
        Xa = feature_matrix(feats, cumulative)
        if Xa.shape[1] <= 1:
            continue
        r = oof_lgbm(Xa, y, folds, **FAST)
        ledger.append({"config": "+" + fam, "ap": r["mean_ap"], "fold_ap": r["fold_ap"], "n_features": int(Xa.shape[1])})
        log(f"ablation +{fam:24s} blocked AP={100 * r['mean_ap']:.2f} ({Xa.shape[1]} features)")
        (ARTIFACTS / "validation_ablation.json").write_text(json.dumps({"ablation": ledger}, indent=2, default=float), encoding="utf-8")
    report["ablation"] = ledger


def _stage_gnn(feats, y, hashes, difficulty, folds, report) -> None:
    try:
        from src.models.gnn import oof_gnn
        edges = pd.read_parquet(ARTIFACTS / "train_edges.parquet")
        prof = pd.read_parquet(ARTIFACTS / "train_wallet_profile.parquet")
        gres = oof_gnn(feats, edges, prof, y, folds)
        report["gnn_blocked"] = {**evaluate(y, gres["oof"], hashes, difficulty), "fold_ap": gres["fold_ap"],
                                 "best_epochs": gres["best_epochs"], "device": gres["device"], "n_params": gres["n_params"],
                                 "gpu": gres["gpu"]}
        log(f"GNN blocked OOF AP={100 * gres['mean_ap']:.2f} folds={[round(100 * a, 2) for a in gres['fold_ap']]} device={gres['device']}")
        pd.DataFrame({"transaction_hash": hashes, "oof_gnn": gres["oof"]}).to_parquet(ARTIFACTS / "oof_gnn.parquet", index=False)
    except Exception as exc:  # the GNN is an experiment; it must never block the submission
        report["gnn_error"] = repr(exc)
        log(f"GNN failed: {exc!r}")
    (ARTIFACTS / "validation_gnn.json").write_text(json.dumps({k: report[k] for k in ("gnn_blocked", "gnn_error") if k in report}, indent=2, default=float), encoding="utf-8")


def _merge_validation(train_dir: Path) -> dict:
    report = json.loads((ARTIFACTS / "validation_lgbm.json").read_text())
    for name in ("validation_ablation.json", "validation_gnn.json", "validation_diagnostics.json"):
        p = ARTIFACTS / name
        if p.exists():
            report.update(json.loads(p.read_text()))
    oof = pd.read_parquet(ARTIFACTS / "oof_lgbm.parquet")
    y = oof["y"].to_numpy()
    hashes = oof["transaction_hash"].to_numpy()
    groups = oof["group"].to_numpy()
    oof_lgbm_scores = oof["oof_lgbm"].to_numpy()
    oof_gnn_scores = None
    if (ARTIFACTS / "oof_gnn.parquet").exists():
        g = pd.read_parquet(ARTIFACTS / "oof_gnn.parquet").set_index("transaction_hash")["oof_gnn"].reindex(hashes)
        if not g.isna().any():
            oof_gnn_scores = g.to_numpy()
    folds = make_folds(groups, y, N_FOLDS, SEED)

    # ---- blend selection on OOF AP -----------------------------------------
    blends = {}
    for w in GNN_BLEND_WEIGHTS:
        if w > 0 and oof_gnn_scores is None:
            continue
        b = blend(oof_lgbm_scores, oof_gnn_scores, w)
        blends[str(w)] = {"ap": float(evaluate(y, b, hashes)["ap"]),
                          "fold_ap": [float(evaluate(y[va], b[va], hashes[va])["ap"]) for _, va in folds]}
    best_w = max(blends, key=lambda k: blends[k]["ap"])
    report["blends"] = blends
    report["chosen_gnn_weight"] = float(best_w)
    report["chosen_blend_ap"] = blends[best_w]["ap"]
    log(f"blend APs: { {k: round(100 * v['ap'], 2) for k, v in blends.items()} } -> chosen GNN weight {best_w}")

    pd.DataFrame({"transaction_hash": hashes, "y": y, "group": groups, "oof_lgbm": oof_lgbm_scores,
                  "oof_gnn": oof_gnn_scores if oof_gnn_scores is not None else np.nan}).to_parquet(ARTIFACTS / "oof_scores.parquet", index=False)
    (ARTIFACTS / "validation.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"wrote {ARTIFACTS / 'validation.json'}")
    return report


# --------------------------------------------------------------------------
# final fit + prediction
# --------------------------------------------------------------------------
def _manifest(train_dir: Path, test_dir: Path, extra: dict) -> None:
    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=train_dir.parents[0], text=True).strip()
    except Exception:
        commit = "unknown"
    import lightgbm, sklearn, numpy, pandas  # noqa
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = None
    files = {}
    for d in (train_dir, test_dir):
        for f in ("transactions.parquet", "transfers.parquet", "market_rates.csv", "labels.csv"):
            if (d / f).exists():
                files[str(d / f)] = sha(d / f)
    man = {"python": sys.version, "platform": platform.platform(), "git_commit": commit, "seed": SEED,
           "versions": {"lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__, "numpy": numpy.__version__,
                        "pandas": pandas.__version__, "torch": torch_v},
           "input_checksums": files,
           "commands": ["python run.py inspect", "python run.py baseline", "python run.py features", "python run.py validate",
                        "python run.py train --final", "python run.py predict --output submission/predictions.csv",
                        "python run.py verify --file submission/predictions.csv"],
           **extra}
    (ARTIFACTS / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")


def train_final(train_dir: Path, test_dir: Path, final: bool = True) -> None:
    feats = build_features_for_split("train", train_dir)
    lab = _labels_for(feats, load_split(train_dir)["labels"])
    y = lab["is_suspicious"].astype(int).to_numpy()
    X = feature_matrix(feats)
    vpath = ARTIFACTS / "validation.json"
    val = json.loads(vpath.read_text()) if vpath.exists() else {}
    n_est = int(val.get("chosen_n_estimators", 600))
    spw = float(val.get("chosen_scale_pos_weight", 1.0))
    gnn_w = float(val.get("chosen_gnn_weight", 0.0))
    log(f"final LightGBM: {X.shape[1]} features, n_estimators={n_est}, scale_pos_weight={spw}")
    model = fit_final(X, y, n_est, spw)
    joblib.dump({"model": model, "columns": list(X.columns), "n_estimators": n_est, "scale_pos_weight": spw}, ARTIFACTS / "lgbm_final.joblib")
    gain_importance(model, X.columns).to_csv(ARTIFACTS / "lgbm_gain_importance.csv", header=["gain"])
    gnn_info = None
    if gnn_w > 0:
        from src.models.gnn import fit_final_gnn
        edges = pd.read_parquet(ARTIFACTS / "train_edges.parquet")
        prof = pd.read_parquet(ARTIFACTS / "train_wallet_profile.parquet")
        epochs = int(np.median(val["gnn_blocked"]["best_epochs"])) if "gnn_blocked" in val else 15
        gnn_info = fit_final_gnn(feats, edges, prof, y, epochs, ARTIFACTS / "gnn_final.pt")
        log(f"final GNN trained for {epochs} epochs on {gnn_info['device']}")
    _manifest(train_dir, test_dir, {"n_features": int(X.shape[1]), "lgbm_n_estimators": n_est, "scale_pos_weight": spw,
                                    "gnn_weight": gnn_w, "gnn": gnn_info, "validation_summary": {
                                        k: val.get(k) for k in ("lgbm_blocked", "gnn_blocked", "blends", "chosen_blend_ap")}})
    log("wrote artifacts/lgbm_final.joblib and manifest.json")


def predict(train_dir: Path, test_dir: Path, output: Path) -> dict:
    feats = build_features_for_split("test", test_dir)
    bundle = joblib.load(ARTIFACTS / "lgbm_final.joblib")
    X = feats.reindex(columns=bundle["columns"]).astype("float32")
    missing = [c for c in bundle["columns"] if c not in feats.columns]
    if missing:
        raise AssertionError(f"test features missing columns: {missing[:10]}")
    lgbm_scores = bundle["model"].predict_proba(X)[:, 1]
    val = json.loads((ARTIFACTS / "validation.json").read_text()) if (ARTIFACTS / "validation.json").exists() else {}
    gnn_w = float(val.get("chosen_gnn_weight", 0.0))
    gnn_scores = None
    if gnn_w > 0 and (ARTIFACTS / "gnn_final.pt").exists():
        from src.models.gnn import predict_gnn
        edges = pd.read_parquet(ARTIFACTS / "test_edges.parquet")
        prof = pd.read_parquet(ARTIFACTS / "test_wallet_profile.parquet")
        gnn_scores = predict_gnn(feats, edges, prof, ARTIFACTS / "gnn_final.pt")
    scores = blend(lgbm_scores, gnn_scores, gnn_w) if gnn_w > 0 else lgbm_scores.astype(float)
    hashes = feats["transaction_hash"]
    tx = pd.read_parquet(test_dir / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
    if len(tx) != EXPECTED_TEST_ROWS:
        log(f"WARNING: test split has {len(tx)} rows, expected {EXPECTED_TEST_ROWS}")
    path = write_predictions(hashes, scores, output)
    v = verify_file(path, tx, expected_rows=len(tx))
    pd.DataFrame({"TRANSACTION_HASH": hashes, "lgbm": lgbm_scores, "gnn": gnn_scores if gnn_scores is not None else np.nan,
                  "score": scores}).to_parquet(ARTIFACTS / "test_scores.parquet", index=False)
    log(f"wrote {path}: {v}")
    if not v["ok"]:
        raise AssertionError(f"submission failed verification: {v['problems']}")
    return v
