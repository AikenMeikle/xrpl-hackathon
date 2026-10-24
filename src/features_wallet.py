"""Wallet-level behaviour projected onto each economic edge.

Column prefixes make the information horizon explicit:

* ``full_``  : statistics over the whole supplied seven-day window
               (allowed: the task is an offline batch investigation).
* ``asof_``  : strictly-before-the-payment windows (would be available in a
               real-time setting).
* ``win_``   : symmetric rolling windows centred on the payment.

``src_``/``dst_`` identify which endpoint the statistic belongs to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EPS, WINDOWS
from src.graph_build import Graph

PROFILE_WINDOWS = {k: v for k, v in WINDOWS.items() if k != "7d"}


def _entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def wallet_profile(g: Graph) -> pd.DataFrame:
    """One row per wallet code: full-window behaviour (no identities)."""
    e = pd.DataFrame({
        "src": g.src, "dst": g.dst, "ts": g.ts,
        "usd": np.nan_to_num(g.usd), "usd_recv": np.nan_to_num(g.usd_recv),
        "paid_asset": g.edges["paid_asset_id"].to_numpy(), "deliv_asset": g.edges["delivered_asset_id"].to_numpy(),
        "conv": g.edges["conversion_flag"].to_numpy(dtype=float),
    })
    n = g.n_wallets
    idx = pd.RangeIndex(n)
    out_g = e.groupby("src")
    in_g = e.groupby("dst")

    prof = pd.DataFrame(index=idx)
    prof["out_tx"] = out_g.size().reindex(idx).fillna(0)
    prof["in_tx"] = in_g.size().reindex(idx).fillna(0)
    prof["out_usd"] = out_g["usd"].sum().reindex(idx).fillna(0)
    prof["in_usd"] = in_g["usd_recv"].sum().reindex(idx).fillna(0)
    prof["out_deg"] = out_g["dst"].nunique().reindex(idx).fillna(0)
    prof["in_deg"] = in_g["src"].nunique().reindex(idx).fillna(0)
    for name, grp, col in (("out", out_g, "usd"), ("in", in_g, "usd_recv")):
        prof[f"{name}_mean"] = grp[col].mean().reindex(idx).fillna(0)
        prof[f"{name}_median"] = grp[col].median().reindex(idx).fillna(0)
        prof[f"{name}_std"] = grp[col].std().reindex(idx).fillna(0)
        prof[f"{name}_max"] = grp[col].max().reindex(idx).fillna(0)
        prof[f"{name}_cv"] = prof[f"{name}_std"] / np.maximum(prof[f"{name}_mean"], EPS)
    prof["out_assets"] = out_g["paid_asset"].nunique().reindex(idx).fillna(0)
    prof["in_assets"] = in_g["deliv_asset"].nunique().reindex(idx).fillna(0)
    prof["out_conversion_frac"] = out_g["conv"].mean().reindex(idx).fillna(0)
    prof["in_conversion_frac"] = in_g["conv"].mean().reindex(idx).fillna(0)
    first = pd.concat([out_g["ts"].min(), in_g["ts"].min()], axis=1).min(axis=1).reindex(idx)
    last = pd.concat([out_g["ts"].max(), in_g["ts"].max()], axis=1).max(axis=1).reindex(idx)
    t_min = int(g.ts.min()) if len(g.ts) else 0
    prof["first_seen_rel_h"] = ((first - t_min) / 3600).fillna(0)
    prof["last_seen_rel_h"] = ((last - t_min) / 3600).fillna(0)
    prof["active_span_h"] = ((last - first) / 3600).fillna(0)
    prof["total_tx"] = prof["out_tx"] + prof["in_tx"]
    prof["tx_per_active_hour"] = prof["total_tx"] / np.maximum(prof["active_span_h"], 1.0)

    # burstiness / temporal entropy over hourly and 10-minute buckets
    all_ev = pd.concat([
        pd.DataFrame({"w": e["src"], "ts": e["ts"], "dir": 0}),
        pd.DataFrame({"w": e["dst"], "ts": e["ts"], "dir": 1}),
    ])
    all_ev["bucket10"] = (all_ev["ts"] - t_min) // 600
    all_ev["hour"] = (all_ev["ts"] - t_min) // 3600
    b10 = all_ev.groupby(["w", "bucket10"]).size()
    prof["max_burst_share_10m"] = (b10.groupby(level=0).max() / b10.groupby(level=0).sum()).reindex(idx).fillna(0)
    hourly = all_ev.groupby(["w", "hour"]).size()
    prof["temporal_entropy_h"] = hourly.groupby(level=0).apply(lambda s: _entropy(s.to_numpy())).reindex(idx).fillna(0)
    out_h = all_ev[all_ev["dir"] == 0].groupby(["w", "hour"]).size()
    in_h = all_ev[all_ev["dir"] == 1].groupby(["w", "hour"]).size()
    prof["out_temporal_entropy_h"] = out_h.groupby(level=0).apply(lambda s: _entropy(s.to_numpy())).reindex(idx).fillna(0)
    prof["in_temporal_entropy_h"] = in_h.groupby(level=0).apply(lambda s: _entropy(s.to_numpy())).reindex(idx).fillna(0)

    # counterparty structure
    pair_out = e.groupby(["src", "dst"]).agg(n=("usd", "size"), usd=("usd", "sum")).reset_index()
    by_src = pair_out.groupby("src")
    prof["one_use_receiver_frac"] = (by_src["n"].apply(lambda s: float((s == 1).mean()))).reindex(idx).fillna(0)
    prof["out_repeat_concentration"] = (by_src["n"].max() / by_src["n"].sum()).reindex(idx).fillna(0)
    prof["out_largest_flow_share"] = (by_src["usd"].max() / np.maximum(by_src["usd"].sum(), EPS)).reindex(idx).fillna(0)
    by_dst = pair_out.groupby("dst")
    prof["one_use_sender_frac"] = (by_dst["n"].apply(lambda s: float((s == 1).mean()))).reindex(idx).fillna(0)
    prof["in_repeat_concentration"] = (by_dst["n"].max() / by_dst["n"].sum()).reindex(idx).fillna(0)
    prof["in_largest_flow_share"] = (by_dst["usd"].max() / np.maximum(by_dst["usd"].sum(), EPS)).reindex(idx).fillna(0)
    senders = {k: set(v) for k, v in pair_out.groupby("dst")["src"]}
    receivers = {k: set(v) for k, v in pair_out.groupby("src")["dst"]}
    disj = np.zeros(n)
    for w in range(n):
        a, b = senders.get(w, set()), receivers.get(w, set())
        u = a | b
        disj[w] = 1 - len(a & b) / len(u) if u else 0.0
    prof["counterparty_disjointness"] = disj

    # flow balance
    i_usd, o_usd = prof["in_usd"].to_numpy(), prof["out_usd"].to_numpy()
    prof["in_out_usd_ratio"] = np.log1p(i_usd) - np.log1p(o_usd)
    prof["pass_through"] = np.minimum(i_usd, o_usd) / np.maximum(np.maximum(i_usd, o_usd), EPS)
    prof["flow_mismatch"] = np.abs(i_usd - o_usd) / (i_usd + o_usd + EPS)
    prof["retained_usd"] = i_usd - o_usd
    prof["deg_ratio"] = np.log1p(prof["in_deg"]) - np.log1p(prof["out_deg"])
    prof["hub_likeness"] = np.log1p(prof["in_deg"] + prof["out_deg"]) * np.log1p(prof["active_span_h"])
    prof["service_likeness"] = prof["hub_likeness"] * (1 - prof["max_burst_share_10m"])

    # inter-event gap regularity
    for name, key in (("out", "src"), ("in", "dst")):
        s = e.sort_values([key, "ts"])
        gap = s.groupby(key)["ts"].diff()
        gs = gap.groupby(s[key])
        prof[f"{name}_gap_cv"] = (gs.std() / gs.mean().replace(0, np.nan)).reindex(idx).fillna(0)
        prof[f"{name}_gap_median_h"] = (gs.median() / 3600).reindex(idx).fillna(0)
    return prof.astype(float)


def neighbour_aggregates(g: Graph, prof: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of profile metrics over 1-hop (and 2-hop of the mean) counterparties."""
    pairs = pd.DataFrame({"a": np.concatenate([g.src, g.dst]), "b": np.concatenate([g.dst, g.src])}).drop_duplicates()
    nb = pairs.merge(prof[cols], left_on="b", right_index=True)
    hop1 = nb.groupby("a")[cols].mean().reindex(prof.index).fillna(0)
    hop1.columns = [f"nb1_{c}" for c in cols]
    nb2 = pairs.merge(hop1, left_on="b", right_index=True)
    hop2 = nb2.groupby("a")[hop1.columns].mean().reindex(prof.index).fillna(0)
    hop2.columns = [c.replace("nb1_", "nb2_") for c in hop1.columns]
    return pd.concat([hop1, hop2], axis=1)


def window_features(g: Graph) -> pd.DataFrame:
    """Rolling-window activity around each edge, both endpoints, both directions."""
    f = pd.DataFrame(index=g.edges.index)
    s, d, ts = g.src, g.dst, g.ts
    oi, ii = g.out_index, g.in_index
    usd = np.nan_to_num(g.usd)
    for name, w in PROFILE_WINDOWS.items():
        for tag, (back, fwd) in {"asof": (w, 0), "win": (w, w)}.items():
            # sender outgoing
            n, v, lo, hi = oi.counts_and_value(s, ts, back, fwd)
            f[f"{tag}_src_out_n_{name}"] = n
            f[f"{tag}_src_out_usd_{name}"] = v
            f[f"{tag}_src_out_deg_{name}"] = oi.distinct(lo, hi)
            # sender incoming
            n, v, lo, hi = ii.counts_and_value(s, ts, back, fwd)
            f[f"{tag}_src_in_n_{name}"] = n
            f[f"{tag}_src_in_usd_{name}"] = v
            f[f"{tag}_src_in_deg_{name}"] = ii.distinct(lo, hi)
            # receiver incoming
            n, v, lo, hi = ii.counts_and_value(d, ts, back, fwd)
            f[f"{tag}_dst_in_n_{name}"] = n
            f[f"{tag}_dst_in_usd_{name}"] = v
            f[f"{tag}_dst_in_deg_{name}"] = ii.distinct(lo, hi)
            # receiver outgoing
            n, v, lo, hi = oi.counts_and_value(d, ts, back, fwd)
            f[f"{tag}_dst_out_n_{name}"] = n
            f[f"{tag}_dst_out_usd_{name}"] = v
            f[f"{tag}_dst_out_deg_{name}"] = oi.distinct(lo, hi)
        # forward-only views for the receiver (does the money leave?) and sender (was it just funded?)
        n, v, lo, hi = oi.counts_and_value(d, ts, 0, w)
        f[f"fwd_dst_out_n_{name}"] = n
        f[f"fwd_dst_out_usd_{name}"] = v
        f[f"fwd_dst_out_deg_{name}"] = oi.distinct(lo, hi)
        f[f"fwd_dst_out_to_in_ratio_{name}"] = v / np.maximum(f[f"asof_dst_in_usd_{name}"] + np.nan_to_num(g.usd_recv), EPS)
        f[f"fwd_dst_best_amount_gap_{name}"] = oi.best_amount_match(lo, hi, usd)
        n, v, lo, hi = ii.counts_and_value(s, ts, w, 0)
        f[f"asof_src_funded_ratio_{name}"] = v / np.maximum(usd, EPS)
        f[f"asof_src_best_amount_gap_{name}"] = ii.best_amount_match(lo, hi, usd)
        f[f"win_src_out_share_{name}"] = usd / np.maximum(f[f"win_src_out_usd_{name}"], EPS)
        f[f"win_dst_in_share_{name}"] = np.nan_to_num(g.usd_recv) / np.maximum(f[f"win_dst_in_usd_{name}"], EPS)
        f[f"win_src_out_distinctness_{name}"] = f[f"win_src_out_deg_{name}"] / np.maximum(f[f"win_src_out_n_{name}"], 1)
        f[f"win_dst_in_distinctness_{name}"] = f[f"win_dst_in_deg_{name}"] / np.maximum(f[f"win_dst_in_n_{name}"], 1)
    # timing
    f["full_dst_secs_to_next_out"] = np.minimum(oi.nearest_gap(d, ts, 0, 7 * 86400, forward=True), 7 * 86400)
    f["asof_src_secs_since_last_in"] = np.minimum(ii.nearest_gap(s, ts, 7 * 86400, 0, forward=False), 7 * 86400)
    f["asof_src_secs_since_last_out"] = np.minimum(oi.nearest_gap(s, ts - 1, 7 * 86400, 0, forward=False), 7 * 86400)
    f["asof_dst_secs_since_last_in"] = np.minimum(ii.nearest_gap(d, ts - 1, 7 * 86400, 0, forward=False), 7 * 86400)
    f["full_src_secs_to_next_out"] = np.minimum(oi.nearest_gap(s, ts + 1, 0, 7 * 86400, forward=True), 7 * 86400)
    f["full_log_dst_secs_to_next_out"] = np.log1p(f["full_dst_secs_to_next_out"])
    f["asof_log_src_secs_since_last_in"] = np.log1p(f["asof_src_secs_since_last_in"])
    f["full_u_turn_1h"] = ((f["full_dst_secs_to_next_out"] < 3600) & (f["asof_src_secs_since_last_in"] < 3600)).astype(float)
    return f


PROJECT_COLS = [
    "out_tx", "in_tx", "out_usd", "in_usd", "out_deg", "in_deg", "out_mean", "in_mean", "out_median", "in_median",
    "out_cv", "in_cv", "out_max", "in_max", "out_assets", "in_assets", "out_conversion_frac", "in_conversion_frac",
    "first_seen_rel_h", "last_seen_rel_h", "active_span_h", "total_tx", "tx_per_active_hour", "max_burst_share_10m",
    "temporal_entropy_h", "out_temporal_entropy_h", "in_temporal_entropy_h", "one_use_receiver_frac",
    "out_repeat_concentration", "out_largest_flow_share", "one_use_sender_frac", "in_repeat_concentration",
    "in_largest_flow_share", "counterparty_disjointness", "in_out_usd_ratio", "pass_through", "flow_mismatch",
    "retained_usd", "deg_ratio", "hub_likeness", "service_likeness", "out_gap_cv", "in_gap_cv", "out_gap_median_h",
    "in_gap_median_h",
]
NB_COLS = ["in_deg", "out_deg", "pass_through", "total_tx", "one_use_sender_frac", "one_use_receiver_frac",
           "hub_likeness", "active_span_h", "counterparty_disjointness"]


def wallet_features(g: Graph, views: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(edge_features, wallet_profile)``."""
    prof = wallet_profile(g)
    prof["component_size"] = np.bincount(views["component"])[views["component"]]
    prof["community_size"] = np.bincount(views["community"])[views["community"]]
    prof["core"] = views["core"]
    prof["clustering"] = views["clustering"]
    prof["pagerank"] = views["pagerank"] * g.n_wallets
    nb = neighbour_aggregates(g, prof, NB_COLS)
    prof = pd.concat([prof, nb], axis=1)

    cols = PROJECT_COLS + ["component_size", "community_size", "core", "clustering", "pagerank"] + list(nb.columns)
    f = pd.DataFrame(index=g.edges.index)
    sp = prof.loc[g.src, cols].to_numpy()
    dp = prof.loc[g.dst, cols].to_numpy()
    for j, c in enumerate(cols):
        f[f"full_src_{c}"] = sp[:, j]
        f[f"full_dst_{c}"] = dp[:, j]
    for c in ("in_usd", "out_usd", "in_deg", "out_deg", "total_tx", "active_span_h", "pass_through", "hub_likeness"):
        a, b = prof.loc[g.src, c].to_numpy(), prof.loc[g.dst, c].to_numpy()
        f[f"full_diff_log_{c}"] = np.log1p(np.clip(a, 0, None)) - np.log1p(np.clip(b, 0, None))
    f["full_same_community"] = (views["community"][g.src] == views["community"][g.dst]).astype(float)

    # pair history
    pair = pd.DataFrame({"s": g.src, "d": g.dst, "usd": np.nan_to_num(g.usd)})
    pc = pair.groupby(["s", "d"])["usd"].agg(["size", "sum"])
    key = pd.MultiIndex.from_arrays([g.src, g.dst])
    f["full_pair_tx"] = pc["size"].reindex(key).to_numpy()
    f["full_pair_usd"] = pc["sum"].reindex(key).to_numpy()
    rev = pd.MultiIndex.from_arrays([g.dst, g.src])
    f["full_pair_reverse_tx"] = pc["size"].reindex(rev).fillna(0).to_numpy()
    f["full_pair_share_of_sender_usd"] = f["full_pair_usd"] / np.maximum(f["full_src_out_usd"], EPS)
    f["full_pair_share_of_receiver_usd"] = f["full_pair_usd"] / np.maximum(f["full_dst_in_usd"], EPS)
    f["full_pair_is_recurring"] = (f["full_pair_tx"] >= 3).astype(float)
    f["full_edge_share_of_pair_usd"] = np.nan_to_num(g.usd) / np.maximum(f["full_pair_usd"], EPS)

    wf = window_features(g)
    out = pd.concat([f, wf], axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
    return out, prof
