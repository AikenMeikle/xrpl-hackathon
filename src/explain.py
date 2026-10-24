"""Judge-facing evidence: SHAP drivers, witness paths, case studies, feature dictionary."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DOCS, SUBMISSION
from src.models.lgbm import shap_contributions
from src.normalize import load_split

FAMILY_DOC = {
    "tx_": ("transaction", "Per-transaction amounts (log USD), legs, conversion spread, fee, UTC time, symbol flags, roundness, amount relative to wallet/asset medians."),
    "full_": ("wallet full-window", "Seven-day wallet behaviour projected on sender (src) / receiver (dst): counts, USD, degrees, entropy, one-use counterparties, pass-through, hub-likeness, pair history, 1/2-hop neighbour means."),
    "asof_": ("wallet as-of", "Strictly-before-the-payment rolling windows (10m/1h/6h/24h) for both endpoints; usable in real time."),
    "win_": ("wallet centred window", "Symmetric rolling windows around the payment (burst detection in both directions)."),
    "fwd_": ("wallet forward window", "What the receiver does in the window after the payment (forwarding, best amount match)."),
    "match_": ("value-aware matching", "Best / batch soft USD match of this payment with the receiver's later outgoing and the sender's earlier incoming payments: quality=exp(-|log(out/in)|/0.2), delays, counts."),
    "fanin_": ("fan-in", "Collector sweep ratio, exit share, deposit clustering, one-use senders; exit-role support counts."),
    "fanout_": ("fan-out", "Payout count/CV/entropy/equal-size fraction, just-funded ratio, recipient forwarding and one-use fractions; funding-edge projection."),
    "sg_": ("scatter-gather", "Distinct time-ordered middlemen between the same source and destination, branch hold times, value match, middleman activity; bounded 3-hop generalisation."),
    "gs_": ("gather-scatter", "6h hub windows: in/out degree, flow mismatch, pass-through, sender/receiver disjointness, in-to-out lag, exits-after-funding."),
    "cycle_": ("cycle", "Time-respecting value-matched simple cycles of 2-5 hops (24h / 7d): counts, shortest return, best match, elapsed, returned/sent, asset switches."),
    "chain_": ("layered stack", "Longest value-matched time-respecting chain through the edge, single-use intermediary fraction, hop delays, retention, peel ratio."),
    "bip_": ("bipartite", "2x2 payer/receiver rectangles (full, hub-free, same-day), candidate block size/density, external contact, synchronous activity."),
    "irr_": ("irregular network", "Day-window ego graph: nodes/edges/density, branching & merging nodes, 2-5 hop route counts, local cycle support, community volume concentration, bridge status, complexity."),
    "graph_context_missing": ("context", "1 when the transaction is not an economic edge (failed / no legs)."),
}


def feature_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        fam, desc = next(((k, v) for k, v in FAMILY_DOC.items() if c.startswith(k)), ("other", ("other", "")))
        horizon = "as-of" if c.startswith(("asof_", "tx_")) else "full seven-day window"
        rows.append({"feature": c, "family": desc[0] if isinstance(desc, tuple) else fam, "horizon": horizon,
                     "description": desc[1] if isinstance(desc, tuple) else ""})
    return pd.DataFrame(rows)


def _summ(row: pd.Series, prefix: str) -> dict:
    keys = ["out_tx", "in_tx", "out_usd", "in_usd", "out_deg", "in_deg", "active_span_h", "pass_through", "one_use_sender_frac", "hub_likeness"]
    return {k: float(row.get(f"full_{prefix}_{k}", np.nan)) for k in keys}


def build_evidence(feats: pd.DataFrame, canonical: pd.DataFrame, scores: np.ndarray, model, columns: list[str],
                   witnesses: dict, top_n: int = 400, extra_hashes: list[str] | None = None) -> list[dict]:
    order = np.argsort(-scores)[:top_n]
    sel = set(order.tolist())
    if extra_hashes:
        hmap = {h: i for i, h in enumerate(feats["transaction_hash"])}
        sel |= {hmap[h] for h in extra_hashes if h in hmap}
    idx = sorted(sel)
    X = feats.iloc[idx][columns].astype("float32")
    contrib = shap_contributions(model, X)
    can = canonical.set_index("transaction_hash")
    out = []
    for r, i in enumerate(idx):
        h = feats["transaction_hash"].iat[i]
        c = can.loc[h]
        top = np.argsort(-np.abs(contrib[r, :-1]))[:8]
        drivers = [{"feature": columns[j], "value": float(X.iat[r, j]), "shap": float(contrib[r, j])} for j in top]
        row = feats.iloc[i]
        motifs = {
            "fan_in": float(row.get("fanin_sweep_quality", 0)), "fan_out": float(row.get("fanout_payout_distinct_1h", 0)),
            "scatter_gather_middlemen": float(row.get("sg_middlemen_count", 0)), "gather_scatter_score_6h": float(row.get("gs_dst_score_6h", 0)),
            "cycle_count": float(row.get("cycle_count_total", 0)), "chain_longest_hops": float(row.get("chain_longest_hops", 0)),
            "bipartite_rectangles": float(row.get("bip_rect_count_nohub", 0)), "irregular_complexity": float(row.get("irr_complexity", 0)),
            "forward_match_q": float(row.get("match_fwd_best_q_24h", 0)), "backward_match_q": float(row.get("match_bwd_best_q_24h", 0)),
        }
        out.append({
            "transaction_hash": h, "score": float(scores[i]), "rank": int(np.where(np.argsort(-scores) == i)[0][0]) + 1,
            "timestamp_utc": str(c["timestamp_utc"]), "source_wallet": c["source_wallet"], "destination_wallet": c["destination_wallet"],
            "paid": {"asset": c["paid_asset_id"], "quantity": float(c["paid_quantity"]), "usd": float(c["paid_usd"])},
            "delivered": {"asset": c["delivered_asset_id"], "quantity": float(c["delivered_quantity"]), "usd": float(c["delivered_usd"])},
            "flags": {"conversion": bool(c["conversion_flag"]), "legs": int(c["transfer_leg_count"]), "partial_payment": bool(c["partial_payment_flag"]),
                      "missing_rate": bool(c["paid_rate_missing"] or c["delivered_rate_missing"]), "fee_xrp": float(c["fee_xrp"])},
            "top_drivers": drivers, "motifs": motifs, "witnesses": witnesses.get(h, {}),
            "source_summary": _summ(row, "src"), "destination_summary": _summ(row, "dst"),
            "model_version": "lgbm_final + optional gnn blend (see artifacts/manifest.json)",
            "status": "flagged for review (synthetic data; no identity or criminality is asserted)",
        })
    return out


def explain(train_dir: Path, test_dir: Path) -> None:
    DOCS.mkdir(exist_ok=True)
    bundle = joblib.load(ARTIFACTS / "lgbm_final.joblib")
    model, columns = bundle["model"], bundle["columns"]
    feature_dictionary(columns).to_csv(ARTIFACTS / "feature_dictionary.csv", index=False)

    # ---- test alerts -------------------------------------------------------
    feats_te = pd.read_parquet(ARTIFACTS / "test_features.parquet")
    can_te = pd.read_parquet(ARTIFACTS / "test_canonical.parquet")
    wit_te = json.loads((ARTIFACTS / "test_witnesses.json").read_text(encoding="utf-8")) if (ARTIFACTS / "test_witnesses.json").exists() else {}
    sc = pd.read_parquet(ARTIFACTS / "test_scores.parquet").set_index("TRANSACTION_HASH")["score"].reindex(feats_te["transaction_hash"]).to_numpy()
    ev = build_evidence(feats_te, can_te, sc, model, columns, wit_te, top_n=400)
    (ARTIFACTS / "test_alert_evidence.json").write_text(json.dumps(ev, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([{"transaction_hash": e["transaction_hash"], "score": e["score"], "rank": e["rank"],
                   "top_driver": e["top_drivers"][0]["feature"], **{f"motif_{k}": v for k, v in e["motifs"].items()}} for e in ev]).to_csv(ARTIFACTS / "test_alert_evidence.csv", index=False)

    # ---- labelled validation: false positives and benign-in-interesting-graph ----
    oof = pd.read_parquet(ARTIFACTS / "oof_scores.parquet")
    feats_tr = pd.read_parquet(ARTIFACTS / "train_features.parquet")
    can_tr = pd.read_parquet(ARTIFACTS / "train_canonical.parquet")
    wit_tr = json.loads((ARTIFACTS / "train_witnesses.json").read_text(encoding="utf-8")) if (ARTIFACTS / "train_witnesses.json").exists() else {}
    labels = load_split(train_dir)["labels"].set_index("TRANSACTION_HASH")
    oof = oof.set_index("transaction_hash").reindex(feats_tr["transaction_hash"])
    s_tr = oof["oof_lgbm"].to_numpy()
    y = oof["y"].to_numpy()
    fp = feats_tr["transaction_hash"].to_numpy()[np.argsort(-np.where(y == 0, s_tr, -1))[:3]].tolist()
    busy = feats_tr["irr_complexity"].fillna(0).to_numpy() if "irr_complexity" in feats_tr else np.zeros(len(y))
    benign_busy = feats_tr["transaction_hash"].to_numpy()[np.argsort(-np.where((y == 0) & (s_tr < np.quantile(s_tr, 0.5)), busy, -1))[:2]].tolist()
    hard_caught = feats_tr["transaction_hash"].to_numpy()[np.argsort(-np.where((y == 1), s_tr, -1))[:3]].tolist()
    ev_tr = build_evidence(feats_tr, can_tr, s_tr, model, columns, wit_tr, top_n=0, extra_hashes=fp + benign_busy + hard_caught)
    for e in ev_tr:
        lab = labels.loc[e["transaction_hash"]]
        e["label"] = {"is_suspicious": int(lab["is_suspicious"]), "family": str(lab.get("family")), "difficulty": str(lab.get("difficulty"))}
    (ARTIFACTS / "train_case_evidence.json").write_text(json.dumps(ev_tr, indent=1, default=str), encoding="utf-8")

    # ---- case studies markdown ------------------------------------------------
    val = json.loads((ARTIFACTS / "validation.json").read_text()) if (ARTIFACTS / "validation.json").exists() else {}
    lines = ["# Case studies (synthetic ledger; every item is 'flagged for review', never an accusation)", ""]
    lines += ["All wallets and hashes are organiser simulation identifiers. Scores are rankings, not probabilities of guilt.", ""]
    # pick 3 complementary shapes from the top test alerts
    picked, seen = [], set()
    for e in ev:
        m = e["motifs"]
        shape = max(("fan_in", "fan_out", "scatter_gather_middlemen", "gather_scatter_score_6h", "cycle_count", "chain_longest_hops", "bipartite_rectangles", "irregular_complexity"),
                    key=lambda k: m[k] / {"fan_in": 1, "fan_out": 10, "scatter_gather_middlemen": 4, "gather_scatter_score_6h": 10, "cycle_count": 2, "chain_longest_hops": 5, "bipartite_rectangles": 20, "irregular_complexity": 6}[k])
        if shape not in seen:
            seen.add(shape); picked.append((shape, e))
        if len(picked) == 3:
            break
    for shape, e in picked:
        lines += [f"## Test alert rank {e['rank']} (dominant shape: {shape})", "",
                  f"- hash `{e['transaction_hash']}` at {e['timestamp_utc']}, score {e['score']:.4f}",
                  f"- {e['source_wallet']} -> {e['destination_wallet']}: paid {e['paid']['quantity']:.4f} {e['paid']['asset'].split(':')[-1]} (~${e['paid']['usd']:.2f}), delivered {e['delivered']['quantity']:.4f} {e['delivered']['asset'].split(':')[-1]} (~${e['delivered']['usd']:.2f}); legs={e['flags']['legs']} conversion={e['flags']['conversion']}",
                  "- top SHAP drivers: " + ", ".join(f"`{d['feature']}`={d['value']:.3g} ({d['shap']:+.2f})" for d in e["top_drivers"][:5]),
                  "- motif evidence: " + ", ".join(f"{k}={v:.3g}" for k, v in e["motifs"].items() if v),
                  "- witnesses: " + (json.dumps(e["witnesses"], default=str)[:600] if e["witnesses"] else "none recorded"), ""]
    lines += ["## Labelled validation: highest-scoring legitimate transactions (false positives)", ""]
    for e in ev_tr:
        if e["label"]["is_suspicious"] == 0 and e["transaction_hash"] in fp:
            lines += [f"- `{e['transaction_hash']}` OOF score {e['score']:.4f}, family `{e['label']['family']}`: drivers " +
                      ", ".join(f"`{d['feature']}`" for d in e["top_drivers"][:4]) +
                      f"; sender tx={e['source_summary']['out_tx']:.0f}/{e['source_summary']['in_tx']:.0f}, receiver pass-through={e['destination_summary']['pass_through']:.2f}. "
                      "Likely a payroll/exchange/merchant look-alike: uniform or rapid flows without purposeless hops."]
    lines += ["", "## Benign-looking transactions inside busy irregular neighbourhoods (low score, high ego complexity)", ""]
    for e in ev_tr:
        if e["transaction_hash"] in benign_busy:
            lines += [f"- `{e['transaction_hash']}` OOF score {e['score']:.4f}, irregular complexity {e['motifs']['irregular_complexity']:.2f}, label {e['label']['family']}: the model keeps it low because value-matching and single-use-intermediary evidence is absent."]
    lines += ["", "## Highest-ranked planted cases (labelled)", ""]
    for e in ev_tr:
        if e["transaction_hash"] in hard_caught:
            lines += [f"- `{e['transaction_hash']}` family `{e['label']['family']}` ({e['label']['difficulty']}), OOF score {e['score']:.4f}: " + ", ".join(f"`{d['feature']}`" for d in e["top_drivers"][:4])]
    lines += ["", "## Why legitimate activity imitates laundering shapes", "",
              "- Exchange / payment-processor hot wallets look like gather-scatter hubs: many senders, many receivers, balanced flow. The counter-evidence is sustained multi-day activity, repeat counterparties and low synchronisation (`full_*_service_likeness`, `full_*_one_use_*_frac`).",
              "- Payroll and refunds look like fan-out: uniform payouts, but recipients keep the money (`fanout_recipient_forward_frac_*` near 0) and the payer was not just funded.",
              "- Merchant intake and fundraisers look like fan-in: many senders, but no prompt sweep of the collected total (`fanin_sweep_quality`, `fanin_last_deposit_to_exit_sec`).",
              "- Market making looks like cycles: value returns, but usually at different scale or with many unrelated flows in between.",
              "", "## Measured validation (blocked flow-group folds)", "",
              f"- LightGBM blocked OOF AP: {100 * val.get('lgbm_blocked', {}).get('ap', float('nan')):.2f} (folds {[round(100 * a, 2) for a in val.get('lgbm_blocked', {}).get('fold_ap', [])]})",
              f"- GNN blocked OOF AP: {100 * val.get('gnn_blocked', {}).get('ap', float('nan')):.2f}" if "gnn_blocked" in val else "- GNN: not available",
              f"- Chosen GNN blend weight: {val.get('chosen_gnn_weight')} (blend AP {100 * val.get('chosen_blend_ap', float('nan')):.2f})", ""]
    (DOCS / "CASE_STUDIES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS / 'CASE_STUDIES.md'}, {ARTIFACTS / 'test_alert_evidence.json'}, {ARTIFACTS / 'feature_dictionary.csv'}")
