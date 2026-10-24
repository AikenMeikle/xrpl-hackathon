"""Canonical transaction table and transfer-leg audit table.

XRPL semantics enforced here:

* XRP amounts on the wire and in ``SOURCE_VALUE`` are integer drops;
  1 XRP = 1,000,000 drops. Issued tokens are never divided.
* An asset is ``issuer:currency``. XRP uses the sentinel address.
* ``SendMax`` / ``Amount`` / ``DeliverMax`` are ceilings. What the sender paid
  is the first executed transfer leg; what arrived is ``DELIVERED_AMOUNT``.
* A 2- or 3-leg conversion is ONE economic payment. Liquidity-provider legs
  stay in the audit table only.
* Fees are burned XRP, never a recipient.
* Failed transactions keep a canonical row (they must be scored) but are not
  economic edges.
* USD uses the exact UTC hour containing the payment; a missing rate stays
  NaN and is flagged, never filled with zero or a future price.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DROPS_PER_XRP, MICRO_USD, XRP_ASSET_ID, XRP_RATE_KEY, XRP_SENTINEL


# --------------------------------------------------------------------------
# small pure helpers (unit-tested)
# --------------------------------------------------------------------------
def asset_id(token_address: str, symbol: str) -> str:
    if symbol == "XRP" or token_address == XRP_SENTINEL:
        return XRP_ASSET_ID
    if not token_address or not symbol:
        raise ValueError(f"asset is missing issuer or symbol: {token_address!r} {symbol!r}")
    return f"{token_address}:{symbol}"


def rate_key(asset: str) -> str:
    return XRP_RATE_KEY if asset in (XRP_ASSET_ID, "XRP") else asset


def rate_hour(timestamp_unix: int) -> int:
    return int(timestamp_unix) // 3600 * 3600


def drops_to_xrp(drops) -> float:
    return int(drops) / DROPS_PER_XRP


def quantity_from_source_value(source_value, token_address: str, symbol: str) -> float:
    """XRP SOURCE_VALUE is drops; token SOURCE_VALUE is already token units."""
    raw = float(source_value)
    if symbol == "XRP" or token_address == XRP_SENTINEL:
        return raw / DROPS_PER_XRP
    return raw


def parse_amount(raw) -> tuple[str, float]:
    """Decode an XRPL amount (JSON text, drops string or IOU dict) to (asset_id, units)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raise ValueError("amount is missing")
    parsed = json.loads(raw) if isinstance(raw, str) and raw.strip()[:1] in ('"', "{") else raw
    if isinstance(parsed, str):
        return XRP_ASSET_ID, drops_to_xrp(parsed)
    if isinstance(parsed, (int, np.integer)):
        return XRP_ASSET_ID, drops_to_xrp(parsed)
    if isinstance(parsed, dict):
        return asset_id(str(parsed["issuer"]), str(parsed["currency"])), float(Decimal(str(parsed["value"])))
    raise ValueError(f"unrecognised amount: {raw!r}")


def load_rate_index(rates: pd.DataFrame) -> dict[tuple[int, str], float]:
    return {
        (int(h), str(a)): int(m) / MICRO_USD
        for h, a, m in zip(rates["TIMESTAMP"], rates["ASSET"], rates["PRICE_MICRO_USD"])
    }


def price_usd(rate_index: dict, timestamp_unix: int, asset: str) -> float:
    """Exact hour lookup. Returns NaN when no rate exists for that hour."""
    return rate_index.get((rate_hour(timestamp_unix), rate_key(asset)), float("nan"))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_split(directory: str | Path) -> dict:
    d = Path(directory)
    out = {
        "dir": d,
        "transactions": pd.read_parquet(d / "transactions.parquet"),
        "transfers": pd.read_parquet(d / "transfers.parquet"),
        "rates": pd.read_csv(d / "market_rates.csv"),
        "labels": pd.read_csv(d / "labels.csv") if (d / "labels.csv").exists() else None,
    }
    for col in ("LEDGER_INDEX", "TIMESTAMP", "TRANSACTION_INDEX", "SOURCE_FEE"):
        if col in out["transactions"].columns:
            out["transactions"][col] = out["transactions"][col].astype("int64")
    for col in ("LEDGER_INDEX", "TIMESTAMP", "TRANSACTION_INDEX", "TRANSFER_INDEX"):
        if col in out["transfers"].columns:
            out["transfers"][col] = out["transfers"][col].astype("int64")
    return out


# --------------------------------------------------------------------------
# canonical tables
# --------------------------------------------------------------------------
CANONICAL_ORDER = ["ledger_index", "transaction_index", "transaction_hash"]


def normalize(transactions: pd.DataFrame, transfers: pd.DataFrame, rates: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(canonical, legs)``.

    ``canonical`` has exactly one row per transaction hash, in ledger order,
    with ``in_economic_graph`` marking rows that form Graph A edges.
    ``legs`` is the raw transfer audit table (Graph B) with asset ids,
    quantities in whole units and USD values per leg.
    """
    tx = transactions.copy()
    if tx["TRANSACTION_HASH"].duplicated().any():
        raise AssertionError("transactions.parquet has duplicate TRANSACTION_HASH")
    rate_index = load_rate_index(rates)

    # ---- legs (Graph B) ------------------------------------------------
    tr = transfers.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"], kind="mergesort").reset_index(drop=True)
    tok = tr["TOKEN_ADDRESS"].astype(str).to_numpy()
    sym = tr["SYMBOL"].astype(str).to_numpy()
    is_xrp = (sym == "XRP") | (tok == XRP_SENTINEL)
    leg_asset = np.where(is_xrp, XRP_ASSET_ID, np.char.add(np.char.add(tok, ":"), sym))
    src_val = tr["SOURCE_VALUE"].astype(float).to_numpy()
    qty = np.where(is_xrp, src_val / DROPS_PER_XRP, src_val)
    # VALUE is the organiser's human-unit column; cross-check only.
    val_col = tr["VALUE"].astype(float).to_numpy()
    value_disagrees = ~np.isclose(qty, val_col, rtol=1e-6, atol=1e-6)
    leg_ts = tr["TIMESTAMP"].astype("int64").to_numpy()
    leg_price = np.array([price_usd(rate_index, t, a) for t, a in zip(leg_ts, leg_asset)])
    legs = pd.DataFrame({
        "transaction_hash": tr["TRANSACTION_HASH"].astype(str),
        "transfer_index": tr["TRANSFER_INDEX"].astype("int64"),
        "ledger_index": tr["LEDGER_INDEX"].astype("int64"),
        "transaction_index": tr["TRANSACTION_INDEX"].astype("int64"),
        "timestamp_unix": leg_ts,
        "from_address": tr["FROM_ADDRESS"].astype(str),
        "to_address": tr["TO_ADDRESS"].astype(str),
        "asset_id": leg_asset,
        "quantity": qty,
        "source_value_raw": src_val,
        "value_column_disagrees": value_disagrees,
        "transfer_type": tr["TRANSFER_TYPE"].astype(str),
        "price_usd": leg_price,
        "usd": qty * leg_price,
    })

    # ---- per-transaction JSON fields ------------------------------------
    extras = [json.loads(s) if isinstance(s, str) else (s or {}) for s in tx["EXTRA_TRANSACTION_FIELDS"]]
    dest = [e.get("Destination") for e in extras]
    dest_tag = [e.get("DestinationTag") for e in extras]
    has_sendmax = np.array(["SendMax" in e for e in extras])
    has_paths = np.array(["Paths" in e for e in extras])

    sendmax_qty = np.full(len(tx), np.nan)
    sendmax_asset = np.array([""] * len(tx), dtype=object)
    ceiling_qty = np.full(len(tx), np.nan)
    ceiling_asset = np.array([""] * len(tx), dtype=object)
    for i, e in enumerate(extras):
        if "SendMax" in e:
            a, q = parse_amount(e["SendMax"])
            sendmax_asset[i], sendmax_qty[i] = a, q
        ceiling = e.get("DeliverMax", e.get("Amount"))
        if ceiling is not None:
            a, q = parse_amount(ceiling)
            ceiling_asset[i], ceiling_qty[i] = a, q

    delivered_asset = np.array([""] * len(tx), dtype=object)
    delivered_qty = np.full(len(tx), np.nan)
    for i, raw in enumerate(tx["DELIVERED_AMOUNT"]):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        a, q = parse_amount(raw)
        delivered_asset[i], delivered_qty[i] = a, q

    success = (tx["TRANSACTION_RESULT"].astype(str) == "tesSUCCESS").to_numpy() & tx["SUCCESS"].astype(bool).to_numpy()
    fee_drops = tx["SOURCE_FEE"].astype("int64").to_numpy()
    ts = tx["TIMESTAMP"].astype("int64").to_numpy()

    # ---- first / last leg per transaction ------------------------------
    g = legs.groupby("transaction_hash", sort=False)
    first = g.head(1).set_index("transaction_hash")
    last = g.tail(1).set_index("transaction_hash")
    n_legs = g.size()
    hashes = tx["TRANSACTION_HASH"].astype(str)

    first_r = first.reindex(hashes)
    last_r = last.reindex(hashes)
    leg_count = n_legs.reindex(hashes).fillna(0).astype(int).to_numpy()

    canonical = pd.DataFrame({
        "transaction_hash": hashes.to_numpy(),
        "timestamp_unix": ts,
        "ledger_index": tx["LEDGER_INDEX"].astype("int64").to_numpy(),
        "transaction_index": tx["TRANSACTION_INDEX"].astype("int64").to_numpy(),
        "source_wallet": tx["ACCOUNT"].astype(str).to_numpy(),
        "destination_wallet": np.array([str(d) if d is not None else "" for d in dest], dtype=object),
        "destination_tag_present": np.array([t is not None for t in dest_tag]),
        "transaction_type": tx["TRANSACTION_TYPE"].astype(str).to_numpy(),
        "transaction_result": tx["TRANSACTION_RESULT"].astype(str).to_numpy(),
        "success": success,
        "fee_drops": fee_drops,
        "fee_xrp": fee_drops / DROPS_PER_XRP,
        "transfer_leg_count": leg_count,
        "paid_asset_id": first_r["asset_id"].to_numpy(),
        "paid_quantity": first_r["quantity"].to_numpy(dtype=float),
        "paid_usd": first_r["usd"].to_numpy(dtype=float),
        "first_leg_from": first_r["from_address"].to_numpy(),
        "last_leg_to": last_r["to_address"].to_numpy(),
        "last_leg_asset_id": last_r["asset_id"].to_numpy(),
        "last_leg_quantity": last_r["quantity"].to_numpy(dtype=float),
        "delivered_asset_id": delivered_asset,
        "delivered_quantity": delivered_qty,
        "ceiling_asset_id": ceiling_asset,
        "ceiling_quantity": ceiling_qty,
        "sendmax_present": has_sendmax,
        "sendmax_quantity": sendmax_qty,
        "paths_present": has_paths,
    })
    canonical["timestamp_utc"] = pd.to_datetime(canonical["timestamp_unix"], unit="s", utc=True)
    canonical["delivered_price_usd"] = [
        price_usd(rate_index, t, a) if a else np.nan
        for t, a in zip(canonical["timestamp_unix"], canonical["delivered_asset_id"])
    ]
    canonical["delivered_usd"] = canonical["delivered_quantity"] * canonical["delivered_price_usd"]
    canonical["paid_rate_missing"] = canonical["paid_usd"].isna() & (leg_count > 0)
    canonical["delivered_rate_missing"] = canonical["delivered_usd"].isna() & (canonical["delivered_asset_id"] != "")
    canonical["conversion_flag"] = (canonical["paid_asset_id"].fillna("") != canonical["delivered_asset_id"]) & (leg_count > 0)
    canonical["conversion_spread"] = (
        (canonical["paid_usd"] - canonical["delivered_usd"]).abs()
        / np.maximum(canonical["paid_usd"].fillna(0.0), 1e-9)
    )
    # Partial payment diagnostic: delivered less than the stated ceiling in the same asset.
    same_asset = canonical["ceiling_asset_id"] == canonical["delivered_asset_id"]
    canonical["partial_payment_flag"] = same_asset & (
        canonical["delivered_quantity"] < canonical["ceiling_quantity"] * (1 - 1e-9)
    )
    canonical["delivered_matches_last_leg"] = np.isclose(
        canonical["delivered_quantity"].fillna(-1), canonical["last_leg_quantity"].fillna(-2), rtol=1e-9, atol=1e-9
    ) & (canonical["delivered_asset_id"] == canonical["last_leg_asset_id"].fillna(""))
    canonical["in_economic_graph"] = success & (leg_count > 0) & (canonical["destination_wallet"] != "")

    canonical = canonical.sort_values(CANONICAL_ORDER, kind="mergesort").reset_index(drop=True)
    _assert_canonical(canonical, transactions)
    return canonical, legs


def _assert_canonical(canonical: pd.DataFrame, transactions: pd.DataFrame) -> None:
    if len(canonical) != len(transactions):
        raise AssertionError("canonical table must have one row per transaction")
    if canonical["transaction_hash"].duplicated().any():
        raise AssertionError("duplicate transaction hash in canonical table")
    edges = canonical[canonical["in_economic_graph"]]
    if (edges["first_leg_from"] != edges["source_wallet"]).any():
        bad = int((edges["first_leg_from"] != edges["source_wallet"]).sum())
        raise AssertionError(f"{bad} edges whose first leg is not from the signing account")
    if (edges["last_leg_to"] != edges["destination_wallet"]).any():
        bad = int((edges["last_leg_to"] != edges["destination_wallet"]).sum())
        raise AssertionError(f"{bad} edges whose last leg does not pay the stated Destination")


def economic_edges(canonical: pd.DataFrame) -> pd.DataFrame:
    """Graph A: one directed edge per successful economic payment, ledger ordered."""
    e = canonical[canonical["in_economic_graph"]].copy().reset_index(drop=True)
    e = e.sort_values(CANONICAL_ORDER, kind="mergesort").reset_index(drop=True)
    return e
