"""Inspection report of the real data package. Writes JSON + Markdown to artifacts/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import ARTIFACTS
from src.normalize import load_split, normalize


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_split(name: str, directory: Path) -> dict:
    d = Path(directory)
    rep: dict = {"name": name, "dir": str(d), "files": {}}
    for f in ("transactions.parquet", "transfers.parquet", "blocks.parquet", "market_rates.csv", "labels.csv"):
        p = d / f
        if p.exists():
            entry = {"bytes": p.stat().st_size, "sha256": sha256(p)}
            if p.suffix == ".parquet":
                schema = pq.read_schema(p)
                entry["rows"] = pq.ParquetFile(p).metadata.num_rows
                entry["schema"] = [f"{x.name}:{x.type}" for x in schema]
            rep["files"][f] = entry
    split = load_split(d)
    tx, tr, rates = split["transactions"], split["transfers"], split["rates"]
    canonical, legs = normalize(tx, tr, rates)
    extras = [json.loads(s) for s in tx["EXTRA_TRANSACTION_FIELDS"]]
    rep.update({
        "n_transactions": int(len(tx)),
        "n_transfers": int(len(tr)),
        "duplicate_hashes": int(tx["TRANSACTION_HASH"].duplicated().sum()),
        "join_key": "TRANSACTION_HASH",
        "time_range_utc": [str(canonical["timestamp_utc"].min()), str(canonical["timestamp_utc"].max())],
        "span_days": float((canonical["timestamp_unix"].max() - canonical["timestamp_unix"].min()) / 86400),
        "transaction_types": tx["TRANSACTION_TYPE"].value_counts().to_dict(),
        "transaction_results": tx["TRANSACTION_RESULT"].value_counts().to_dict(),
        "n_failed": int((~canonical["success"]).sum()),
        "legs_per_transaction": {int(k): int(v) for k, v in legs.groupby("transaction_hash").size().value_counts().sort_index().items()},
        "transfer_types": tr["TRANSFER_TYPE"].value_counts().to_dict(),
        "assets": {f"{k[0]}:{k[1]}": int(v) for k, v in tr.groupby(["TOKEN_ADDRESS", "SYMBOL"]).size().items()},
        "extra_field_key_sets": {" ,".join(k): int(v) for k, v in pd.Series([tuple(sorted(e.keys())) for e in extras]).value_counts().items()},
        "delivered_amount_examples": tx["DELIVERED_AMOUNT"].head(2).tolist(),
        "fee_drops": {"min": int(canonical["fee_drops"].min()), "max": int(canonical["fee_drops"].max()), "mean": float(canonical["fee_drops"].mean())},
        "wallets": int(pd.concat([tr["FROM_ADDRESS"], tr["TO_ADDRESS"]]).nunique()),
        "senders": int(tx["ACCOUNT"].nunique()),
        "economic_edges": int(canonical["in_economic_graph"].sum()),
        "conversions": int(canonical["conversion_flag"].sum()),
        "paid_rate_missing": int(canonical["paid_rate_missing"].sum()),
        "delivered_rate_missing": int(canonical["delivered_rate_missing"].sum()),
        "value_column_disagreements": int(legs["value_column_disagrees"].sum()),
        "delivered_matches_last_leg": int(canonical["delivered_matches_last_leg"].sum()),
        "partial_payment_flags": int(canonical["partial_payment_flag"].sum()),
        "sendmax_present": int(canonical["sendmax_present"].sum()),
        "destination_tag_present": int(canonical["destination_tag_present"].sum()),
        "conversion_spread_quantiles": {str(q): float(v) for q, v in canonical.loc[canonical["conversion_flag"], "conversion_spread"].quantile([0.5, 0.9, 0.99, 1.0]).items()},
        "paid_usd_quantiles": {str(q): float(v) for q, v in canonical["paid_usd"].quantile([0.01, 0.5, 0.99, 1.0]).items()},
        "rates": {"rows": int(len(rates)), "columns": rates.columns.tolist(), "assets": rates["ASSET"].unique().tolist(),
                  "hours": int(rates["TIMESTAMP"].nunique())},
        "max_same_timestamp": int(canonical["timestamp_unix"].value_counts().max()),
        "ledgers": int(canonical["ledger_index"].nunique()),
    })
    if split["labels"] is not None:
        lab = split["labels"]
        rep["labels"] = {
            "columns": lab.columns.tolist(),
            "rows": int(len(lab)),
            "positives": int(lab["is_suspicious"].sum()),
            "prevalence": float(lab["is_suspicious"].mean()),
            "family_by_difficulty": {k: v for k, v in pd.crosstab(lab["family"].fillna("none"), lab["difficulty"].fillna("none")).to_dict().items()},
            "hash_coverage": int(lab["TRANSACTION_HASH"].isin(tx["TRANSACTION_HASH"]).sum()),
        }
    return rep


def write_report(train_dir: Path, test_dir: Path) -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    report = {"train": inspect_split("train", train_dir), "test": inspect_split("test", test_dir)}
    tr_w = set(pd.read_parquet(Path(train_dir) / "transfers.parquet")["FROM_ADDRESS"])
    te_w = set(pd.read_parquet(Path(test_dir) / "transfers.parquet")["FROM_ADDRESS"])
    tr_h = set(pd.read_parquet(Path(train_dir) / "transactions.parquet")["TRANSACTION_HASH"])
    te_h = set(pd.read_parquet(Path(test_dir) / "transactions.parquet")["TRANSACTION_HASH"])
    report["cross_split"] = {"shared_wallets": len(tr_w & te_w), "shared_hashes": len(tr_h & te_h)}
    (ARTIFACTS / "inspection_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Inspection report", ""]
    for k in ("train", "test"):
        r = report[k]
        lines += [f"## {k} ({r['dir']})", ""]
        for key in ("n_transactions", "n_transfers", "duplicate_hashes", "time_range_utc", "span_days", "transaction_types",
                    "transaction_results", "n_failed", "legs_per_transaction", "transfer_types", "assets",
                    "extra_field_key_sets", "fee_drops", "wallets", "senders", "economic_edges", "conversions",
                    "paid_rate_missing", "delivered_rate_missing", "value_column_disagreements",
                    "delivered_matches_last_leg", "partial_payment_flags", "sendmax_present",
                    "destination_tag_present", "conversion_spread_quantiles", "paid_usd_quantiles", "rates",
                    "max_same_timestamp", "ledgers", "labels"):
            if key in r:
                lines.append(f"- **{key}**: `{json.dumps(r[key], default=str)}`")
        lines.append("")
    lines += ["## cross split", f"- `{json.dumps(report['cross_split'])}`", ""]
    (ARTIFACTS / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
