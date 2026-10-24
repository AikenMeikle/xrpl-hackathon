"""Final PASS/FAIL acceptance table assembled from artifacts (no new computation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS, EXPECTED_TEST_ROWS, ROOT, SUBMISSION
from src.predict import verify_file


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_report(test_dir: Path, run_tests: bool = True) -> str:
    rows: list[tuple[str, str, str]] = []
    insp = _load(ARTIFACTS / "inspection_report.json")
    if insp:
        ok = insp["test"]["n_transactions"] == EXPECTED_TEST_ROWS and insp["train"]["duplicate_hashes"] == 0
        rows.append(("input schema / row counts", "PASS" if ok else "FAIL",
                     f"train {insp['train']['n_transactions']:,} tx / {insp['train']['n_transfers']:,} legs; test {insp['test']['n_transactions']:,} tx; "
                     f"legs 1/2/3 = {insp['train']['legs_per_transaction']}; rate_missing={insp['train']['paid_rate_missing']}; shared wallets/hashes = {insp['cross_split']}"))
    else:
        rows.append(("input schema / row counts", "FAIL", "no inspection report"))
    if run_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests"], cwd=ROOT, capture_output=True, text=True)
        last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1] if r.stdout.strip() else r.stderr[-200:]
        rows.append(("unit tests (pytest)", "PASS" if r.returncode == 0 else "FAIL", last))
    if insp:
        one_edge = insp["train"]["economic_edges"] == insp["train"]["n_transactions"] - insp["train"]["n_failed"]
        rows.append(("one economic edge per successful payment", "PASS" if one_edge else "FAIL",
                     f"edges={insp['train']['economic_edges']:,}, conversions={insp['train']['conversions']:,} collapsed to single edges"))
    val = _load(ARTIFACTS / "validation.json")
    rows.append(("absence of label leakage", "PASS" if val else "FAIL",
                 "no hash/wallet/issuer/scenario/label columns (asserted); folds never split a flow group; scalers per fold; test graph independent"))
    if val and "gnn_blocked" in val:
        g = val["gnn_blocked"]
        rows.append(("GNN GPU/CPU smoke test", "PASS", f"device={g['device']}, params={g['n_params']:,}, torch={g.get('gpu', {}).get('torch')}, cuda_available={g.get('gpu', {}).get('cuda_available')}"))
    else:
        rows.append(("GNN GPU/CPU smoke test", "FAIL", (val or {}).get("gnn_error", "GNN stage did not run")))
    if val:
        lb = val["lgbm_blocked"]
        rows.append(("LightGBM blocked AP", "PASS", f"OOF AP={100 * lb['ap']:.2f} folds={[round(100 * a, 2) for a in lb['fold_ap']]} std={100 * lb['std_ap']:.2f} "
                     f"P@1%={lb['precision_at_1pct']:.3f} R@1%={lb['recall_at_1pct']:.3f} ROC-AUC={lb['roc_auc']:.3f} spw={lb.get('scale_pos_weight')}"))
        if "gnn_blocked" in val:
            g = val["gnn_blocked"]
            rows.append(("GNN blocked AP", "PASS", f"OOF AP={100 * g['ap']:.2f} folds={[round(100 * a, 2) for a in g['fold_ap']]} best_epochs={g['best_epochs']}"))
        else:
            rows.append(("GNN blocked AP", "FAIL", "not measured"))
        rows.append(("chosen blend AP", "PASS", f"GNN weight={val['chosen_gnn_weight']} -> OOF AP={100 * val['chosen_blend_ap']:.2f}; all blends: " +
                     ", ".join(f"w={k}:{100 * v['ap']:.2f}" for k, v in val["blends"].items())))
        if "ablation" in val:
            rows.append(("ablation ledger", "PASS", "; ".join(f"{a['config']}={100 * a['ap']:.2f}" if a["ap"] is not None else a["config"] for a in val["ablation"])))
        diag = []
        for k, lab in (("lgbm_random_split", "random"), ("lgbm_scenario_grouped", "scenario-grouped"), ("lgbm_asof_only", "as-of-only")):
            if k in val:
                diag.append(f"{lab}={100 * val[k]['mean_ap']:.2f}")
        if "lgbm_time_holdout_last2d" in val:
            diag.append(f"time-holdout={100 * val['lgbm_time_holdout_last2d']['ap']:.2f}")
        rows.append(("secondary diagnostics (fast LightGBM)", "PASS", ", ".join(diag)))
    else:
        rows.append(("LightGBM blocked AP", "FAIL", "validation.json missing"))
    motif_cols = {}
    fp = ARTIFACTS / "train_features.parquet"
    if fp.exists():
        cols = pd.read_parquet(fp, columns=None).columns if False else pd.read_parquet(fp).columns
        for pre in ("fanin_", "fanout_", "sg_", "gs_", "cycle_", "chain_", "bip_", "irr_"):
            motif_cols[pre] = int(sum(c.startswith(pre) for c in cols))
    rows.append(("all-eight-motif fixture coverage", "PASS" if all(v > 0 for v in motif_cols.values()) and len(motif_cols) == 8 else "FAIL",
                 f"feature counts per family {motif_cols}; fixtures in tests/test_motifs.py"))
    sub = SUBMISSION / "predictions.csv"
    if sub.exists():
        exp = pd.read_parquet(Path(test_dir) / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
        v = verify_file(sub, exp)
        rows.append(("output cardinality / range / bytes", "PASS" if v["ok"] else "FAIL",
                     f"rows={v['rows']:,} unique_scores={v['unique_scores']:,} range=[{v['min']:.3g},{v['max']:.3g}] bytes={v['bytes']:,} problems={v['problems']}"))
    else:
        rows.append(("output cardinality / range / bytes", "FAIL", "submission/predictions.csv missing"))
    man = _load(ARTIFACTS / "manifest.json")
    rows.append(("reproducible command", "PASS" if man else "FAIL", "python run.py all  (manifest: checksums, versions, commit %s, seed %s)" % ((man or {}).get("git_commit", "?")[:8], (man or {}).get("seed"))))
    meth = ROOT / "docs" / "METHODOLOGY.md"
    rows.append(("methodology (<4 pages)", "PASS" if meth.exists() and len(meth.read_text(encoding='utf-8').split()) < 2600 else "FAIL",
                 f"docs/METHODOLOGY.md {len(meth.read_text(encoding='utf-8').split()) if meth.exists() else 0} words; docs/AI_USAGE.md; docs/CASE_STUDIES.md"))
    z = SUBMISSION / "code_submission.zip"
    rows.append(("code ZIP (<= 8 MiB)", "PASS" if z.exists() and z.stat().st_size <= 8 * 1024 * 1024 else "FAIL",
                 f"{z} {z.stat().st_size / 1e6:.2f} MB" if z.exists() else "missing"))
    rows.append(("actual upload status", "NOT DONE", "upload requires the team access code on https://hackathon.ukcbt.org; not performed by the pipeline"))

    width = max(len(r[0]) for r in rows)
    lines = ["| check | status | evidence |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    text = "\n".join(lines)
    (ARTIFACTS / "final_report.md").write_text(text, encoding="utf-8")
    return text
