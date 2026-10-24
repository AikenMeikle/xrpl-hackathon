"""Per-transaction features from the canonical table (no graph context needed).

Every row of the canonical table gets a row here, including failed or
non-payment transactions. Issuer addresses differ between splits, so asset
identity enters only through split-stable symbols (XRP/USD/EUR/GBP) and
relative-to-asset statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EPS


def _symbol(asset_id: pd.Series) -> pd.Series:
    return asset_id.fillna("").astype(str).str.split(":").str[-1]


def _roundness(q: np.ndarray, base: float) -> np.ndarray:
    q = np.nan_to_num(q)
    return ((q > 0) & (np.abs(q / base - np.round(q / base)) < 1e-9)).astype(float)


def transaction_features(canonical: pd.DataFrame) -> pd.DataFrame:
    c = canonical
    f = pd.DataFrame(index=c.index)
    paid = c["paid_usd"].to_numpy(dtype=float)
    deliv = c["delivered_usd"].to_numpy(dtype=float)
    f["tx_log_paid_usd"] = np.log1p(np.clip(np.nan_to_num(paid), 0, None))
    f["tx_log_delivered_usd"] = np.log1p(np.clip(np.nan_to_num(deliv), 0, None))
    f["tx_paid_usd_missing"] = c["paid_rate_missing"].astype(float)
    f["tx_delivered_usd_missing"] = c["delivered_rate_missing"].astype(float)
    f["tx_log_paid_quantity"] = np.log1p(np.clip(np.nan_to_num(c["paid_quantity"].to_numpy(dtype=float)), 0, None))
    f["tx_log_delivered_quantity"] = np.log1p(np.clip(np.nan_to_num(c["delivered_quantity"].to_numpy(dtype=float)), 0, None))
    f["tx_conversion_flag"] = c["conversion_flag"].astype(float)
    f["tx_transfer_leg_count"] = c["transfer_leg_count"].astype(float)
    f["tx_conversion_spread"] = np.where(c["conversion_flag"], c["conversion_spread"].to_numpy(dtype=float), 0.0)
    f["tx_value_retention"] = np.where(paid > 0, np.minimum(deliv, paid) / np.maximum(np.maximum(deliv, paid), EPS), np.nan)
    f["tx_fee_drops"] = c["fee_drops"].astype(float)
    f["tx_fee_to_usd"] = c["fee_xrp"].to_numpy(dtype=float) / np.maximum(np.nan_to_num(paid), EPS)
    f["tx_success"] = c["success"].astype(float)
    f["tx_in_economic_graph"] = c["in_economic_graph"].astype(float)
    f["tx_destination_tag_present"] = c["destination_tag_present"].astype(float)
    f["tx_sendmax_present"] = c["sendmax_present"].astype(float)
    f["tx_paths_present"] = c["paths_present"].astype(float)
    f["tx_partial_payment"] = c["partial_payment_flag"].astype(float)
    f["tx_self_payment"] = (c["source_wallet"] == c["destination_wallet"]).astype(float)
    sm = c["sendmax_quantity"].to_numpy(dtype=float)
    pq = c["paid_quantity"].to_numpy(dtype=float)
    f["tx_sendmax_headroom"] = np.where(np.isfinite(sm) & (sm > 0), pq / np.maximum(sm, EPS), np.nan)
    f["tx_ceiling_ratio"] = np.where(
        c["ceiling_quantity"].to_numpy(dtype=float) > 0,
        c["delivered_quantity"].to_numpy(dtype=float) / np.maximum(c["ceiling_quantity"].to_numpy(dtype=float), EPS), np.nan)

    # time of day (UTC)
    ts = c["timestamp_unix"].to_numpy(dtype=np.int64)
    f["tx_hour_of_day"] = ((ts // 3600) % 24).astype(float)
    f["tx_day_of_week"] = (((ts // 86400) + 4) % 7).astype(float)  # 1970-01-01 was a Thursday
    f["tx_is_night"] = ((f["tx_hour_of_day"] < 6) | (f["tx_hour_of_day"] >= 22)).astype(float)
    f["tx_day_index"] = ((ts - ts.min()) // 86400).astype(float)
    f["tx_minute_of_hour"] = ((ts // 60) % 60).astype(float)

    # split-stable asset identity via symbol only
    ps, ds = _symbol(c["paid_asset_id"]), _symbol(c["delivered_asset_id"])
    for sym in ("XRP", "USD", "EUR", "GBP"):
        f[f"tx_paid_is_{sym}"] = (ps == sym).astype(float)
        f[f"tx_delivered_is_{sym}"] = (ds == sym).astype(float)
    f["tx_same_symbol_different_issuer"] = ((ps == ds) & (c["paid_asset_id"].fillna("") != c["delivered_asset_id"].fillna("")) & (ps != "")).astype(float)

    # roundness (weak signal)
    pqty = c["paid_quantity"].to_numpy(dtype=float)
    dqty = c["delivered_quantity"].to_numpy(dtype=float)
    for base in (10, 100, 1000):
        f[f"tx_paid_round_{base}"] = _roundness(pqty, base)
        f[f"tx_delivered_round_{base}"] = _roundness(dqty, base)
    f["tx_paid_has_fraction"] = ((np.abs(np.nan_to_num(pqty) - np.round(np.nan_to_num(pqty))) > 1e-9)).astype(float)
    f["tx_paid_usd_round_100"] = _roundness(np.round(np.nan_to_num(paid), 2), 100)

    # amount quantiles within split, relative-to-wallet and relative-to-asset
    s_paid = pd.Series(paid)
    f["tx_paid_usd_quantile"] = s_paid.rank(pct=True).to_numpy()
    grp_sender_med = s_paid.groupby(c["source_wallet"].to_numpy()).transform("median").to_numpy()
    grp_recv_med = pd.Series(deliv).groupby(c["destination_wallet"].to_numpy()).transform("median").to_numpy()
    grp_asset_med = s_paid.groupby(c["paid_asset_id"].fillna("").to_numpy()).transform("median").to_numpy()
    f["tx_log_paid_vs_sender_median"] = np.log((np.nan_to_num(paid) + 1) / (np.nan_to_num(grp_sender_med) + 1))
    f["tx_log_delivered_vs_receiver_median"] = np.log((np.nan_to_num(deliv) + 1) / (np.nan_to_num(grp_recv_med) + 1))
    f["tx_log_paid_vs_asset_median"] = np.log((np.nan_to_num(paid) + 1) / (np.nan_to_num(grp_asset_med) + 1))
    sender_n = c.groupby("source_wallet")["transaction_hash"].transform("size").to_numpy(dtype=float)
    recv_n = c.groupby("destination_wallet")["transaction_hash"].transform("size").to_numpy(dtype=float)
    f["tx_sender_activity_scarcity"] = 1.0 / sender_n
    f["tx_receiver_activity_scarcity"] = 1.0 / np.maximum(recv_n, 1)
    f["tx_log_sender_tx_count"] = np.log1p(sender_n)
    f["tx_log_receiver_tx_count"] = np.log1p(recv_n)
    return f.astype("float32")
