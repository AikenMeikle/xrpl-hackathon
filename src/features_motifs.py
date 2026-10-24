"""Value-aware temporal matching engine and the eight typology detectors.

Everything here is label-free and attaches to the PARTICULAR economic edge.
Each family emits counts, best soft match quality, lags / spans, USD scale,
topology size and a witness list of transaction hashes for the best match.

Prefixes: ``match_`` (forwarding engine), ``fanin_``, ``fanout_``, ``sg_``
(scatter-gather), ``gs_`` (gather-scatter), ``cycle_``, ``chain_`` (layered
stack), ``bip_`` (bipartite), ``irr_`` (irregular network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DUST_USD, EPS, MAX_CHAIN_HOPS, MAX_CYCLE_HOPS, VALUE_MATCH_SCALE
from src.graph_build import Graph

FWD_WINDOWS = {"1h": 3600, "6h": 6 * 3600, "24h": 86400}
MAX_FWD_CANDIDATES = 64          # closest-in-time outgoing payments considered per edge
PAIR_QUALITY_FLOOR = 0.30        # |log ratio| <= 0.24 ~ within 27% keeps a hop "value matched"
MAX_TABLE_ROWS = 1_500_000
EGO_CAP = 60
HUB_QUANTILE = 0.97


def quality(out_usd, in_usd) -> np.ndarray:
    out_usd = np.maximum(np.asarray(out_usd, dtype=float), EPS)
    in_usd = np.maximum(np.asarray(in_usd, dtype=float), EPS)
    return np.exp(-np.abs(np.log(out_usd / in_usd)) / VALUE_MATCH_SCALE)


def _cv(x: np.ndarray) -> float:
    return float(np.std(x) / (np.mean(x) + EPS)) if len(x) > 1 else 0.0


def _entropy(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    b = np.round(np.log1p(x) * 4).astype(int)
    _, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------
# 1. temporal matching engine + 2-hop pair table
# --------------------------------------------------------------------------
def forward_pairs(g: Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-edge forward/backward matching stats and the time-respecting 2-hop pair table.

    Pair (e1, e2) means e1 = s->m at t1 and e2 = m->d at t2 strictly after e1 in
    ledger order within 7 days. Dust payments are never pair members.
    """
    n = g.n_edges
    oi, ii = g.out_index, g.in_index
    usd_in = np.nan_to_num(g.usd_recv)
    usd_out = np.nan_to_num(g.usd)
    f = {k: np.zeros(n) for k in [
        "match_fwd_best_q_1h", "match_fwd_best_q_6h", "match_fwd_best_q_24h", "match_fwd_delay_best_24h",
        "match_fwd_n_q50_24h", "match_fwd_same_asset_best", "match_fwd_batch_q_1h", "match_fwd_batch_q_6h",
        "match_fwd_batch_q_24h", "match_fwd_batch_n_6h", "match_fwd_cum_out_ratio_6h",
        "match_bwd_best_q_1h", "match_bwd_best_q_6h", "match_bwd_best_q_24h", "match_bwd_delay_best_24h",
        "match_bwd_n_q50_24h", "match_bwd_batch_q_1h", "match_bwd_batch_q_6h", "match_bwd_batch_q_24h",
        "match_bwd_batch_n_6h", "match_bwd_cum_in_ratio_6h", "match_route_q_product",
    ]}
    f["match_fwd_delay_best_24h"][:] = 86400.0
    f["match_bwd_delay_best_24h"][:] = 86400.0
    witness_fwd = np.full(n, -1, dtype=np.int64)
    witness_bwd = np.full(n, -1, dtype=np.int64)
    pe1, pe2, pq, pdt = [], [], [], []
    deliv_asset = g.edges["delivered_asset_id"].to_numpy()
    paid_asset = g.edges["paid_asset_id"].to_numpy()
    week = 7 * 86400
    lo_f, hi_f = oi.bounds(g.dst, g.ts, 0, week)
    lo_b, hi_b = ii.bounds(g.src, g.ts, 86400, 0)
    for e in range(n):
        t = g.ts[e]
        # ---- forward: receiver's later outgoing payments ----
        a, b = int(lo_f[e]), int(hi_f[e])
        if b > a and usd_in[e] >= DUST_USD:
            tt, uu, _cp, ee = oi.slice_arrays(a, b)
            later = (tt > t) | ((tt == t) & (ee > e))
            later &= uu >= DUST_USD
            if later.any():
                tt, uu, ee = tt[later], uu[later], ee[later]
                if len(ee) > MAX_FWD_CANDIDATES:
                    tt, uu, ee = tt[:MAX_FWD_CANDIDATES], uu[:MAX_FWD_CANDIDATES], ee[:MAX_FWD_CANDIDATES]
                q = quality(uu, usd_in[e])
                dt = tt - t
                pe1.append(np.full(len(ee), e)); pe2.append(ee); pq.append(q); pdt.append(dt)
                for name, w in FWD_WINDOWS.items():
                    m = dt <= w
                    if m.any():
                        f[f"match_fwd_best_q_{name}"][e] = q[m].max()
                        cum = np.cumsum(uu[m])
                        k = int(np.searchsorted(cum, usd_in[e] * 0.9))
                        k = min(k, len(cum) - 1)
                        f[f"match_fwd_batch_q_{name}"][e] = float(quality(cum[k], usd_in[e]))
                        if name == "6h":
                            f["match_fwd_batch_n_6h"][e] = k + 1
                            f["match_fwd_cum_out_ratio_6h"][e] = cum[-1] / usd_in[e]
                m24 = dt <= 86400
                if m24.any():
                    j = int(np.argmax(q[m24]))
                    f["match_fwd_delay_best_24h"][e] = dt[m24][j]
                    f["match_fwd_n_q50_24h"][e] = int((q[m24] >= 0.5).sum())
                    best_e = int(ee[m24][j])
                    witness_fwd[e] = best_e
                    f["match_fwd_same_asset_best"][e] = float(paid_asset[best_e] == deliv_asset[e])
        # ---- backward: sender's earlier incoming payments ----
        a, b = int(lo_b[e]), int(hi_b[e])
        if b > a and usd_out[e] >= DUST_USD:
            tt, uu, _cp, ee = ii.slice_arrays(a, b)
            earlier = (tt < t) | ((tt == t) & (ee < e))
            earlier &= uu >= DUST_USD
            if earlier.any():
                tt, uu, ee = tt[earlier], uu[earlier], ee[earlier]
                q = quality(usd_out[e], uu)
                dt = t - tt
                for name, w in FWD_WINDOWS.items():
                    m = dt <= w
                    if m.any():
                        f[f"match_bwd_best_q_{name}"][e] = q[m].max()
                        cum = np.cumsum(uu[m][::-1])  # most recent first
                        k = int(np.searchsorted(cum, usd_out[e] * 0.9))
                        k = min(k, len(cum) - 1)
                        f[f"match_bwd_batch_q_{name}"][e] = float(quality(usd_out[e], cum[k]))
                        if name == "6h":
                            f["match_bwd_batch_n_6h"][e] = k + 1
                            f["match_bwd_cum_in_ratio_6h"][e] = cum[-1] / usd_out[e]
                j = int(np.argmax(q))
                f["match_bwd_delay_best_24h"][e] = dt[j]
                f["match_bwd_n_q50_24h"][e] = int((q >= 0.5).sum())
                witness_bwd[e] = int(ee[j])
    f["match_route_q_product"] = f["match_fwd_best_q_24h"] * f["match_bwd_best_q_24h"]
    f["match_log_fwd_delay"] = np.log1p(f["match_fwd_delay_best_24h"])
    f["match_log_bwd_delay"] = np.log1p(f["match_bwd_delay_best_24h"])
    feats = pd.DataFrame(f, index=g.edges.index)
    feats["match_witness_fwd"] = witness_fwd
    feats["match_witness_bwd"] = witness_bwd
    if pe1:
        pairs = pd.DataFrame({"e1": np.concatenate(pe1), "e2": np.concatenate(pe2),
                              "q": np.concatenate(pq), "dt": np.concatenate(pdt)})
    else:
        pairs = pd.DataFrame({"e1": np.array([], dtype=int), "e2": np.array([], dtype=int), "q": np.array([]), "dt": np.array([])})
    pairs["s"] = g.src[pairs["e1"].to_numpy()] if len(pairs) else np.array([], dtype=int)
    pairs["m"] = g.dst[pairs["e1"].to_numpy()] if len(pairs) else np.array([], dtype=int)
    pairs["d"] = g.dst[pairs["e2"].to_numpy()] if len(pairs) else np.array([], dtype=int)
    return feats, pairs


# --------------------------------------------------------------------------
# 2/3/4. fan-in, fan-out, gather-scatter (window slices around each edge)
# --------------------------------------------------------------------------
def hub_window_features(g: Graph, prof: pd.DataFrame, feats_match: pd.DataFrame) -> pd.DataFrame:
    n = g.n_edges
    oi, ii = g.out_index, g.in_index
    usd_out = np.nan_to_num(g.usd)
    usd_in = np.nan_to_num(g.usd_recv)
    forwards_1h = feats_match["match_fwd_delay_best_24h"].to_numpy() <= 3600
    forwards_24h = feats_match["match_fwd_delay_best_24h"].to_numpy() < 86400
    total_tx = prof["total_tx"].to_numpy()
    one_use = (prof["in_tx"].to_numpy() <= 1) & (prof["out_tx"].to_numpy() <= 1)
    out = {}

    def col(name):
        return out.setdefault(name, np.zeros(n))

    # fan-in: receiver B as collector --------------------------------------
    lo6, hi6 = ii.bounds(g.dst, g.ts, 6 * 3600, 3600)   # deposit cluster may continue up to the exit
    lo1f, hi1f = oi.bounds(g.dst, g.ts, 0, 3600)
    lo10, hi10 = ii.bounds(g.dst, g.ts, 600, 600)
    # fan-in: sender B as exit (this edge is the sweep)
    lo6s, hi6s = ii.bounds(g.src, g.ts, 6 * 3600, 0)
    # fan-out: sender A payouts
    lo_o = {w: oi.bounds(g.src, g.ts, s, s) for w, s in (("10m", 600), ("1h", 3600), ("6h", 21600))}
    lo_in1, hi_in1 = ii.bounds(g.src, g.ts, 3600, 0)
    # funding edge: receiver A fans out within 1h after
    lo_f1, hi_f1 = oi.bounds(g.dst, g.ts, 0, 3600)
    # gather-scatter: 6h centred windows for both endpoints
    gs_in = {role: ii.bounds(w, g.ts, 21600, 21600) for role, w in (("dst", g.dst), ("src", g.src))}
    gs_out = {role: oi.bounds(w, g.ts, 21600, 21600) for role, w in (("dst", g.dst), ("src", g.src))}

    for e in range(n):
        t = g.ts[e]
        # ---- fan-in (deposit role) ----
        tt1, uu1, cp1, ee1 = oi.slice_arrays(int(lo1f[e]), int(hi1f[e]))
        later = tt1 >= t
        exit_usd = uu1[later].sum() if later.any() else 0.0
        largest_exit = uu1[later].max() if later.any() else 0.0
        t_exit = tt1[later][int(np.argmax(uu1[later]))] if later.any() else t
        tt, uu, cp, ee = ii.slice_arrays(int(lo6[e]), int(hi6[e]))
        keep = tt <= t_exit
        tt, uu, cp, ee = tt[keep], uu[keep], cp[keep], ee[keep]
        in_usd6 = uu.sum()
        n_senders6 = len(set(cp.tolist()))
        col("fanin_in_degree_6h")[e] = n_senders6
        col("fanin_preceding_in_usd_6h")[e] = in_usd6
        col("fanin_collector_exit_usd_1h")[e] = exit_usd
        col("fanin_sweep_ratio")[e] = min(exit_usd / (in_usd6 + EPS), 5.0)
        col("fanin_log_sweep_ratio")[e] = np.log1p(exit_usd) - np.log1p(in_usd6)
        col("fanin_largest_exit_share")[e] = min(largest_exit / (in_usd6 + EPS), 5.0)
        col("fanin_sweep_quality")[e] = float(quality(largest_exit, in_usd6)) if largest_exit > 0 else 0.0
        col("fanin_last_deposit_to_exit_sec")[e] = (tt1[later].min() - t) if later.any() else 3600.0
        col("fanin_one_use_sender_frac")[e] = float(one_use[cp].mean()) if len(cp) else 0.0
        t10, _u, c10, _e = ii.slice_arrays(int(lo10[e]), int(hi10[e]))
        col("fanin_clustered_deposit_count_10m")[e] = len(c10)
        col("fanin_clustered_distinct_senders_10m")[e] = len(set(c10.tolist()))
        col("fanin_deposit_burstiness")[e] = len(c10) / max(len(cp), 1)
        col("fanin_edge_support_count")[e] = n_senders6 if exit_usd > 0 else 0.0
        # ---- fan-in (exit role: this edge sweeps what the sender collected) ----
        tt, uu, cp, ee = ii.slice_arrays(int(lo6s[e]), int(hi6s[e]))
        before = tt <= t
        col("fanin_exit_support_senders_6h")[e] = len(set(cp[before].tolist())) if before.any() else 0.0
        col("fanin_exit_sweep_quality")[e] = float(quality(usd_out[e], uu[before].sum())) if before.any() else 0.0
        col("fanin_exit_one_use_sender_frac")[e] = float(one_use[cp[before]].mean()) if before.any() else 0.0
        # ---- fan-out (payout role) ----
        for w, (lo, hi) in lo_o.items():
            tt, uu, cp, ee = oi.slice_arrays(int(lo[e]), int(hi[e]))
            col(f"fanout_payout_n_{w}")[e] = len(uu)
            col(f"fanout_payout_distinct_{w}")[e] = len(set(cp.tolist()))
            col(f"fanout_payout_cv_{w}")[e] = _cv(uu)
            col(f"fanout_equal_size_frac_{w}")[e] = float((np.abs(uu - usd_out[e]) <= 0.05 * usd_out[e]).mean()) if len(uu) else 0.0
            if w == "1h":
                col("fanout_payout_entropy_1h")[e] = _entropy(uu)
                col("fanout_recipient_forward_frac_1h")[e] = float(forwards_1h[ee].mean()) if len(ee) else 0.0
                col("fanout_recipient_forward_frac_24h")[e] = float(forwards_24h[ee].mean()) if len(ee) else 0.0
                col("fanout_recipient_one_use_frac")[e] = float(one_use[cp].mean()) if len(cp) else 0.0
                col("fanout_recipient_low_activity_frac")[e] = float((total_tx[cp] <= 3).mean()) if len(cp) else 0.0
                pay_sum = uu.sum()
            if w == "10m":
                col("fanout_largest_10m_outflow_share")[e] = uu.max() / (uu.sum() + EPS) if len(uu) else 0.0
        ti, ui, ci, ei = ii.slice_arrays(int(lo_in1[e]), int(hi_in1[e]))
        col("fanout_just_funded_ratio_1h")[e] = ui.sum() / (pay_sum + EPS) if len(ui) else 0.0
        col("fanout_source_inflow_to_split_lag")[e] = (t - ti.max()) if len(ti) else 3600.0
        # ---- fan-out (funding role: receiver splits this within 1h) ----
        tt, uu, cp, ee = oi.slice_arrays(int(lo_f1[e]), int(hi_f1[e]))
        later = tt >= t
        col("fanout_funds_split_deg_1h")[e] = len(set(cp[later].tolist())) if later.any() else 0.0
        col("fanout_funds_split_cv_1h")[e] = _cv(uu[later]) if later.sum() > 1 else 0.0
        col("fanout_funds_split_ratio_1h")[e] = min(uu[later].sum() / (usd_in[e] + EPS), 5.0) if later.any() else 0.0
        # ---- gather-scatter (both roles) ----
        for role in ("dst", "src"):
            lo_i, hi_i = gs_in[role]
            lo_o_, hi_o_ = gs_out[role]
            ti, ui, ci, ei = ii.slice_arrays(int(lo_i[e]), int(hi_i[e]))
            to, uo, co, eo = oi.slice_arrays(int(lo_o_[e]), int(hi_o_[e]))
            si, so = set(ci.tolist()), set(co.tolist())
            in_u, out_u = ui.sum(), uo.sum()
            col(f"gs_{role}_in_deg_6h")[e] = len(si)
            col(f"gs_{role}_out_deg_6h")[e] = len(so)
            col(f"gs_{role}_flow_mismatch_6h")[e] = abs(in_u - out_u) / (in_u + out_u + EPS)
            col(f"gs_{role}_pass_through_6h")[e] = min(in_u, out_u) / (max(in_u, out_u) + EPS)
            col(f"gs_{role}_disjointness_6h")[e] = 1 - len(si & so) / len(si | so) if (si or so) else 0.0
            col(f"gs_{role}_both_sides_6h")[e] = np.sqrt(len(si) * len(so))
            col(f"gs_{role}_score_6h")[e] = np.sqrt(len(si) * len(so)) * (min(in_u, out_u) / (max(in_u, out_u) + EPS))
            col(f"gs_{role}_out_amount_cv_6h")[e] = _cv(uo)
            if len(to) and len(ti):
                idx = np.searchsorted(ti, to, side="right") - 1
                ok = idx >= 0
                lags = (to[ok] - ti[idx[ok]]) if ok.any() else np.array([21600.0])
                col(f"gs_{role}_median_in_to_out_lag")[e] = float(np.median(lags))
                col(f"gs_{role}_exits_after_funding_frac_1h")[e] = float((lags <= 3600).mean())
            else:
                col(f"gs_{role}_median_in_to_out_lag")[e] = 21600.0
    return pd.DataFrame(out, index=g.edges.index)


# --------------------------------------------------------------------------
# 5. scatter-gather via the pair table
# --------------------------------------------------------------------------
def scatter_gather(g: Graph, pairs: pd.DataFrame, prof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = g.n_edges
    cols = ["sg_middlemen_count", "sg_branch_count", "sg_branch_holdtime_median_h", "sg_branch_holdtime_cv",
            "sg_value_match_quality", "sg_middleman_low_activity_frac", "sg_middleman_one_in_one_out_frac",
            "sg_route_span_h", "sg_3hop_middle_pairs", "sg_log_usd"]
    out = pd.DataFrame(0.0, index=g.edges.index, columns=cols)
    wit: dict[int, list] = {}
    if len(pairs) == 0:
        return out, wit
    p = pairs[(pairs["s"] != pairs["d"]) & (pairs["dt"] <= 86400)].copy()
    if len(p) == 0:
        return out, wit
    usd1 = np.nan_to_num(g.usd)[p["e1"].to_numpy()]
    usd2 = np.nan_to_num(g.usd)[p["e2"].to_numpy()]
    p["usd1"], p["usd2"] = usd1, usd2
    p["t1"] = g.ts[p["e1"].to_numpy()]
    p["t2"] = g.ts[p["e2"].to_numpy()]
    low = (prof["total_tx"].to_numpy() <= 4)
    oio = (prof["in_tx"].to_numpy() == 1) & (prof["out_tx"].to_numpy() == 1)
    p["low"] = low[p["m"].to_numpy()].astype(float)
    p["oio"] = oio[p["m"].to_numpy()].astype(float)
    # keep one (best-quality) pair per (s, m, d, e1) to count middlemen rather than conversion legs
    grp = p.groupby(["s", "d"])
    agg = grp.agg(mid=("m", "nunique"), branches=("e1", "nunique"), hold_med=("dt", "median"),
                  hold_std=("dt", "std"), hold_mean=("dt", "mean"), u1=("usd1", "sum"), u2=("usd2", "sum"),
                  g_low=("low", "mean"), g_oio=("oio", "mean"), tmin=("t1", "min"), tmax=("t2", "max"))
    agg = agg[agg["mid"] >= 2]
    if len(agg) == 0:
        return out, wit
    agg["gq"] = quality(agg["u2"], agg["u1"])
    agg["hold_cv"] = (agg["hold_std"].fillna(0) / (agg["hold_mean"] + EPS))
    agg["span_h"] = (agg["tmax"] - agg["tmin"]) / 3600
    pj = p.merge(agg, left_on=["s", "d"], right_index=True)
    # 3-hop generalisation: s->m1->m2->d, bounded by joining the pair table on e2 == e1'
    p3 = p[["e1", "e2", "s", "d", "q"]].merge(pairs[["e1", "e2", "d"]].rename(columns={"e1": "e2", "e2": "e3", "d": "d3"}), on="e2")
    p3 = p3[(p3["d3"] != p3["s"])].head(MAX_TABLE_ROWS)
    if len(p3):
        c3 = p3.groupby(["s", "d3"])["d"].nunique()
        c3 = c3[c3 >= 2]
        p3 = p3.merge(c3.rename("mid3"), left_on=["s", "d3"], right_index=True)
        for ecol in ("e1", "e2", "e3"):
            v = p3.groupby(ecol)["mid3"].max()
            out.loc[v.index, "sg_3hop_middle_pairs"] = np.maximum(out.loc[v.index, "sg_3hop_middle_pairs"].to_numpy(), v.to_numpy())
    # project to both edges of every branch: take the strongest (s,d) group per edge
    for ecol in ("e1", "e2"):
        best = pj.sort_values(["mid", "gq"], ascending=False).drop_duplicates(ecol).set_index(ecol)
        idx = best.index.to_numpy()
        out.loc[idx, "sg_middlemen_count"] = np.maximum(out.loc[idx, "sg_middlemen_count"].to_numpy(), best["mid"].to_numpy())
        out.loc[idx, "sg_branch_count"] = np.maximum(out.loc[idx, "sg_branch_count"].to_numpy(), best["branches"].to_numpy())
        out.loc[idx, "sg_branch_holdtime_median_h"] = best["hold_med"].to_numpy() / 3600
        out.loc[idx, "sg_branch_holdtime_cv"] = best["hold_cv"].to_numpy()
        out.loc[idx, "sg_value_match_quality"] = np.maximum(out.loc[idx, "sg_value_match_quality"].to_numpy(), best["gq"].to_numpy())
        out.loc[idx, "sg_middleman_low_activity_frac"] = best["g_low"].to_numpy()
        out.loc[idx, "sg_middleman_one_in_one_out_frac"] = best["g_oio"].to_numpy()
        out.loc[idx, "sg_route_span_h"] = best["span_h"].to_numpy()
        out.loc[idx, "sg_log_usd"] = np.log1p(best["u1"].to_numpy())
    # witnesses for the strongest groups
    top = agg.sort_values(["mid", "gq"], ascending=False).head(200)
    hashes = g.edges["transaction_hash"].to_numpy()
    for (s, d), _row in top.iterrows():
        rows = p[(p["s"] == s) & (p["d"] == d)]
        hs = sorted(set(rows["e1"].tolist()) | set(rows["e2"].tolist()))
        for e in hs:
            wit.setdefault(int(e), [hashes[k] for k in hs[:12]])
    return out, wit


# --------------------------------------------------------------------------
# 6. temporal cycles and 7. layered stacks via value-matched pair joins
# --------------------------------------------------------------------------
def cycles_and_chains(g: Graph, pairs: pd.DataFrame, prof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = g.n_edges
    hashes = g.edges["transaction_hash"].to_numpy()
    ts = g.ts
    usd = np.nan_to_num(g.usd)
    conv = g.edges["conversion_flag"].to_numpy().astype(float)
    paid_asset = g.edges["paid_asset_id"].to_numpy()
    deliv_asset = g.edges["delivered_asset_id"].to_numpy()
    cyc_cols = ["cycle_count_total", "cycle_count_24h", "cycle_shortest_return_hops", "cycle_best_value_match",
                "cycle_best_elapsed_h", "cycle_returned_to_sent_ratio", "cycle_asset_switches_best",
                "cycle_roundtrip_same_asset", "cycle_log_usd_scale", "cycle_edge_participation", "cycle_2hop_count"]
    chain_cols = ["chain_longest_hops", "chain_forward_hops", "chain_backward_hops", "chain_edge_long_chain_count",
                  "chain_single_use_intermediary_frac", "chain_hop_retention_median", "chain_hop_delay_min_h",
                  "chain_hop_delay_median_h", "chain_hop_delay_max_h", "chain_hop_delay_cv", "chain_span_h",
                  "chain_distinct_assets", "chain_branch_scarcity", "chain_peel_ratio_median", "chain_log_usd"]
    out = pd.DataFrame(0.0, index=g.edges.index, columns=cyc_cols + chain_cols)
    out["cycle_shortest_return_hops"] = 0.0
    wit: dict[int, dict] = {}
    if len(pairs) == 0:
        return out, wit
    P = pairs[pairs["q"] >= PAIR_QUALITY_FLOOR][["e1", "e2", "q", "dt", "s", "m", "d"]].reset_index(drop=True)
    if len(P) > MAX_TABLE_ROWS:
        P = P.sort_values("q", ascending=False).head(MAX_TABLE_ROWS).reset_index(drop=True)

    # ---------------- cycles: iterative join, vertices distinct, strictly time ordered ----------------
    part = np.zeros(n)
    best_q = np.zeros(n)
    best_len = np.zeros(n)
    best_elapsed = np.full(n, np.inf)
    best_ret = np.zeros(n)
    best_switch = np.zeros(n)
    best_same = np.zeros(n)
    counts = np.zeros(n)
    counts24 = np.zeros(n)
    shortest = np.full(n, np.inf)
    two_hop = np.zeros(n)
    # paths table: columns e_0..e_k, nodes visited, cumulative q, origin s
    paths = P.rename(columns={"e1": "e0", "e2": "e1", "q": "cq", "dt": "elapsed"})
    paths["v0"] = paths["s"]; paths["v1"] = paths["m"]; paths["v2"] = paths["d"]
    paths = paths.drop(columns=["s", "m", "d"])
    for L in range(2, MAX_CYCLE_HOPS + 1):
        if len(paths) == 0:
            break
        closed = paths[paths[f"v{L}"] == paths["v0"]]
        if len(closed):
            ecols = [f"e{i}" for i in range(L)]
            E = closed[ecols].to_numpy()
            q = closed["cq"].to_numpy()
            el = closed["elapsed"].to_numpy()
            ret = usd[E[:, -1]] / np.maximum(usd[E[:, 0]], EPS)
            sw = conv[E].sum(axis=1)
            same = (paid_asset[E[:, 0]] == deliv_asset[E[:, -1]]).astype(float)
            for e_pos in range(L):
                col_e = E[:, e_pos]
                np.add.at(part, col_e, 1)
                np.add.at(counts, col_e, 1)
                np.add.at(counts24, col_e[el <= 86400], 1)
                shortest[col_e] = np.minimum(shortest[col_e], L)
                if L == 2:
                    np.add.at(two_hop, col_e, 1)
                better = q > best_q[col_e]
                # keep best (first occurrence per edge on ties)
                for i in np.where(better)[0]:
                    e = col_e[i]
                    if q[i] > best_q[e]:
                        best_q[e] = q[i]; best_len[e] = L; best_elapsed[e] = el[i]; best_ret[e] = ret[i]
                        best_switch[e] = sw[i]; best_same[e] = same[i]
                        wit[int(e)] = {"motif": "cycle", "hops": L, "hashes": [hashes[k] for k in E[i]], "quality": float(q[i])}
        if L == MAX_CYCLE_HOPS:
            break
        openp = paths[paths[f"v{L}"] != paths["v0"]]
        if len(openp) == 0:
            break
        nxt = openp.merge(P[["e1", "e2", "q", "dt", "d"]].rename(columns={"e1": f"e{L - 1}", "e2": f"e{L}", "q": "q_new", "dt": "dt_new", "d": f"v{L + 1}"}), on=f"e{L - 1}")
        # no repeated internal vertex (closing on v0 is allowed), elapsed within 7 days
        ok = np.ones(len(nxt), dtype=bool)
        for i in range(1, L + 1):
            ok &= (nxt[f"v{L + 1}"] != nxt[f"v{i}"]).to_numpy()
        nxt = nxt[ok]
        nxt["cq"] = nxt["cq"] * nxt["q_new"]
        nxt["elapsed"] = nxt["elapsed"] + nxt["dt_new"]
        nxt = nxt[nxt["elapsed"] <= 7 * 86400].drop(columns=["q_new", "dt_new"])
        if len(nxt) > MAX_TABLE_ROWS:
            nxt = nxt.sort_values("cq", ascending=False).head(MAX_TABLE_ROWS)
        paths = nxt.reset_index(drop=True)
    out["cycle_count_total"] = counts
    out["cycle_count_24h"] = counts24
    out["cycle_shortest_return_hops"] = np.where(np.isfinite(shortest), shortest, 0.0)
    out["cycle_best_value_match"] = best_q
    out["cycle_best_elapsed_h"] = np.where(np.isfinite(best_elapsed), best_elapsed / 3600, 0.0)
    out["cycle_returned_to_sent_ratio"] = np.minimum(best_ret, 5.0)
    out["cycle_asset_switches_best"] = best_switch
    out["cycle_roundtrip_same_asset"] = best_same
    out["cycle_log_usd_scale"] = np.where(counts > 0, np.log1p(usd), 0.0)
    out["cycle_edge_participation"] = part
    out["cycle_2hop_count"] = two_hop

    # ---------------- layered stack: forward / backward DP over value-matched hops ----------------
    P24 = P[P["dt"] <= 86400]
    succ = P24.groupby("e1")["e2"].apply(list).to_dict()
    succ_q = P24.groupby("e1")["q"].apply(list).to_dict()
    pred = P24.groupby("e2")["e1"].apply(list).to_dict()
    pred_q = P24.groupby("e2")["q"].apply(list).to_dict()
    order = np.argsort(ts, kind="mergesort")  # ledger order == index order already, but be explicit
    F = np.ones(n); Fn = np.zeros(n, dtype=np.int64) - 1; Fc = np.ones(n)
    for e in order[::-1]:
        ss = succ.get(e)
        if not ss:
            continue
        best = 1.0; arg = -1; cnt = 0.0
        for e2, q in zip(ss, succ_q[e]):
            if F[e2] + 1 > best:
                best = F[e2] + 1; arg = e2
            if F[e2] >= 2:
                cnt += Fc[e2]
        F[e] = best; Fn[e] = arg; Fc[e] = max(cnt, 1.0)
    Bk = np.ones(n); Bn = np.zeros(n, dtype=np.int64) - 1; Bc = np.ones(n)
    for e in order:
        pp = pred.get(e)
        if not pp:
            continue
        best = 1.0; arg = -1; cnt = 0.0
        for e1, q in zip(pp, pred_q[e]):
            if Bk[e1] + 1 > best:
                best = Bk[e1] + 1; arg = e1
            if Bk[e1] >= 2:
                cnt += Bc[e1]
        Bk[e] = best; Bn[e] = arg; Bc[e] = max(cnt, 1.0)
    longest = np.minimum(F + Bk - 1, MAX_CHAIN_HOPS + 2)
    out["chain_longest_hops"] = longest
    out["chain_forward_hops"] = F
    out["chain_backward_hops"] = Bk
    out["chain_edge_long_chain_count"] = np.where(longest >= 3, np.minimum(Fc * Bc, 1e4), 0.0)
    single_use = (prof["in_tx"].to_numpy() <= 2) & (prof["out_tx"].to_numpy() <= 2)
    for e in np.where(longest >= 3)[0]:
        chain = []
        x = e
        while Bn[x] >= 0 and len(chain) < MAX_CHAIN_HOPS:
            x = Bn[x]; chain.append(x)
        chain = chain[::-1] + [e]
        x = e
        while Fn[x] >= 0 and len(chain) < MAX_CHAIN_HOPS + 1:
            x = Fn[x]; chain.append(x)
        chain = np.array(chain)
        inter = g.dst[chain[:-1]]
        hop_dt = np.diff(ts[chain]) / 3600
        ratios = usd[chain[1:]] / np.maximum(usd[chain[:-1]], EPS)
        out.iat[e, out.columns.get_loc("chain_single_use_intermediary_frac")] = float(single_use[inter].mean())
        out.iat[e, out.columns.get_loc("chain_hop_retention_median")] = float(np.median(quality(usd[chain[1:]], usd[chain[:-1]])))
        out.iat[e, out.columns.get_loc("chain_hop_delay_min_h")] = float(hop_dt.min())
        out.iat[e, out.columns.get_loc("chain_hop_delay_median_h")] = float(np.median(hop_dt))
        out.iat[e, out.columns.get_loc("chain_hop_delay_max_h")] = float(hop_dt.max())
        out.iat[e, out.columns.get_loc("chain_hop_delay_cv")] = _cv(hop_dt)
        out.iat[e, out.columns.get_loc("chain_span_h")] = float((ts[chain[-1]] - ts[chain[0]]) / 3600)
        out.iat[e, out.columns.get_loc("chain_distinct_assets")] = float(len(set(paid_asset[chain].tolist()) | set(deliv_asset[chain].tolist())))
        out.iat[e, out.columns.get_loc("chain_branch_scarcity")] = float(np.mean(prof["out_deg"].to_numpy()[inter] <= 2))
        out.iat[e, out.columns.get_loc("chain_peel_ratio_median")] = float(np.median(np.minimum(ratios, 2.0)))
        out.iat[e, out.columns.get_loc("chain_log_usd")] = float(np.log1p(usd[chain].mean()))
        if int(e) not in wit or wit[int(e)].get("motif") != "cycle":
            wit[int(e)] = {"motif": "chain", "hops": int(len(chain)), "hashes": [hashes[k] for k in chain]}
    return out, wit


# --------------------------------------------------------------------------
# 8. bipartite blocks and 9. irregular ego networks (dense 700x700 views)
# --------------------------------------------------------------------------
def bipartite_and_irregular(g: Graph, views: dict, prof: pd.DataFrame) -> pd.DataFrame:
    n, W = g.n_edges, g.n_wallets
    ts = g.ts
    usd = np.nan_to_num(g.usd)
    t0 = ts.min()
    day = ((ts - t0) // 86400).astype(int)
    n_days = int(day.max()) + 1
    total_tx = prof["total_tx"].to_numpy()
    hub = (total_tx >= np.quantile(total_tx[total_tx > 0], HUB_QUANTILE)) & (total_tx >= 50)
    M = np.zeros((W, W), dtype=np.int32)
    np.add.at(M, (g.src, g.dst), 1)
    Mday = np.zeros((n_days, W, W), dtype=np.int16)
    np.add.at(Mday, (day, g.src, g.dst), 1)
    Musd = np.zeros((W, W))
    np.add.at(Musd, (g.src, g.dst), usd)

    def rect_counts(B: np.ndarray) -> np.ndarray:
        Bf = B.astype(np.float64)
        C = Bf @ Bf.T                       # shared receivers between payers
        R = C @ Bf                          # R[a,b] = sum_a' C[a,a'] B[a',b]
        rowdeg = Bf.sum(axis=1)
        coldeg = Bf.sum(axis=0)
        rect = R - rowdeg[:, None] * Bf - (coldeg[None, :] - Bf)
        return np.where(Bf > 0, np.maximum(rect, 0), 0), C

    B_full = (M > 0).astype(np.int8)
    B_nohub = B_full.copy()
    B_nohub[hub, :] = 0
    B_nohub[:, hub] = 0
    rect_full, C_full = rect_counts(B_full)
    rect_nohub, C_nohub = rect_counts(B_nohub)
    out = {}
    out["bip_rect_count_full"] = rect_full[g.src, g.dst]
    out["bip_rect_count_nohub"] = rect_nohub[g.src, g.dst]
    out["bip_log_rect_count_nohub"] = np.log1p(out["bip_rect_count_nohub"])
    # per-day rectangles (synchronous blocks)
    rect_day = np.zeros(n)
    for dday in range(n_days):
        Bd = (Mday[dday] > 0).astype(np.int8)
        Bd[hub, :] = 0; Bd[:, hub] = 0
        rd, _ = rect_counts(Bd)
        m = day == dday
        rect_day[m] = rd[g.src[m], g.dst[m]]
    out["bip_rect_count_sameday"] = rect_day
    # block statistics per unique (a,b) pair on the hub-free graph
    rowsets = [np.where(B_nohub[a] > 0)[0] for a in range(W)]
    block_cols = ["bip_payer_shared_receiver_count", "bip_receiver_set_jaccard_max", "bip_block_payers",
                  "bip_block_receivers", "bip_block_density", "bip_external_contact_frac", "bip_block_amount_cv",
                  "bip_sync_pair_activity_1h", "bip_sync_pair_activity_6h"]
    blk = {c: np.zeros(n) for c in block_cols}
    pair_df = pd.DataFrame({"a": g.src, "b": g.dst, "e": np.arange(n)})
    for (a, b), grp in pair_df.groupby(["a", "b"]):
        if hub[a] or hub[b] or B_nohub[a, b] == 0:
            continue
        shared = C_nohub[a].copy(); shared[a] = 0
        co_payers = np.where((shared >= 2) & (B_nohub[:, b] > 0))[0]
        eidx = grp["e"].to_numpy()
        blk["bip_payer_shared_receiver_count"][eidx] = len(co_payers)
        if len(co_payers) == 0:
            continue
        ra = rowsets[a]
        jac = max(len(np.intersect1d(ra, rowsets[a2])) / max(len(np.union1d(ra, rowsets[a2])), 1) for a2 in co_payers)
        A_star = np.concatenate([[a], co_payers])
        recv_counts = B_nohub[A_star].sum(axis=0)
        B_star = np.where(recv_counts >= 2)[0]
        sub = B_nohub[np.ix_(A_star, B_star)]
        obs = sub.sum()
        dens = obs / max(len(A_star) * len(B_star), 1)
        outside = B_nohub[A_star].sum() - obs
        ext = outside / max(B_nohub[A_star].sum(), 1)
        amounts = Musd[np.ix_(A_star, B_star)][sub > 0]
        blk["bip_receiver_set_jaccard_max"][eidx] = jac
        blk["bip_block_payers"][eidx] = len(A_star)
        blk["bip_block_receivers"][eidx] = len(B_star)
        blk["bip_block_density"][eidx] = dens
        blk["bip_external_contact_frac"][eidx] = ext
        blk["bip_block_amount_cv"][eidx] = _cv(amounts) if len(amounts) > 1 else 0.0
        # synchronous activity: other block edges near in time
        in_block = np.isin(g.src, A_star) & np.isin(g.dst, B_star)
        tb = ts[in_block]
        for e in eidx:
            blk["bip_sync_pair_activity_1h"][e] = int((np.abs(tb - ts[e]) <= 3600).sum()) - 1
            blk["bip_sync_pair_activity_6h"][e] = int((np.abs(tb - ts[e]) <= 21600).sum()) - 1
    out.update(blk)

    # ---------------- irregular ego networks on day-window adjacency ----------------
    comm = views["community"]
    core = views["core"]
    pr = views["pagerank"]
    comm_of_edge = comm[g.src]
    internal = comm[g.src] == comm[g.dst]
    comm_usd_total = np.bincount(comm_of_edge, weights=usd, minlength=comm.max() + 1)
    comm_internal = np.bincount(comm_of_edge, weights=usd * internal, minlength=comm.max() + 1)
    comm_in_from_outside = np.bincount(comm[g.dst], weights=usd * (~internal), minlength=comm.max() + 1)
    comm_out_to_outside = np.bincount(comm[g.src], weights=usd * (~internal), minlength=comm.max() + 1)
    c = comm_of_edge
    out["irr_community_internal_frac"] = comm_internal[c] / (comm_usd_total[c] + EPS)
    out["irr_community_boundary_imbalance"] = np.abs(comm_in_from_outside[c] - comm_out_to_outside[c]) / (comm_in_from_outside[c] + comm_out_to_outside[c] + EPS)
    out["irr_bridge_edge"] = ((~internal) & (pr[g.src] + pr[g.dst] > np.quantile(pr, 0.8) * 2)).astype(float)
    out["irr_local_kcore_min"] = np.minimum(core[g.src], core[g.dst])
    # community volume concentration around this edge
    conc6 = np.zeros(n); conc24 = np.zeros(n)
    for cid in np.unique(c):
        m = np.where(c == cid)[0]
        tt = ts[m]; uu = usd[m]
        order = np.argsort(tt); tt_s = tt[order]; cum = np.concatenate([[0.0], np.cumsum(uu[order])])
        tot = cum[-1] + EPS
        for w, arr in ((21600, conc6), (86400, conc24)):
            lo = np.searchsorted(tt_s, tt - w, "left"); hi = np.searchsorted(tt_s, tt + w, "right")
            arr[m] = (cum[hi] - cum[lo]) / tot
    out["irr_community_volume_conc_6h"] = conc6
    out["irr_community_volume_conc_24h"] = conc24

    ego_cols = ["irr_ego_nodes", "irr_ego_edges", "irr_ego_density", "irr_branching_nodes", "irr_merging_nodes",
                "irr_branch_merge_interleaving", "irr_routes_2to5", "irr_route_length_diversity",
                "irr_local_cycle_support", "irr_asset_hop_diversity", "irr_complexity"]
    ego = {k: np.zeros(n) for k in ego_cols}
    asset_pairs = pd.DataFrame({"s": g.src, "d": g.dst, "a": g.edges["paid_asset_id"].to_numpy(), "day": day})
    assets_by_pair_day = asset_pairs.groupby(["day", "s"])["a"].nunique()
    keys = pd.DataFrame({"a": g.src, "b": g.dst, "day": day, "e": np.arange(n)})
    for (a, b, dday), grp in keys.groupby(["a", "b", "day"]):
        lo_d, hi_d = max(dday - 1, 0), min(dday + 1, n_days - 1)
        Aw = Mday[lo_d:hi_d + 1].sum(axis=0)
        nb = np.where((Aw[a] > 0) | (Aw[:, a] > 0) | (Aw[b] > 0) | (Aw[:, b] > 0))[0]
        nb = nb[(nb != a) & (nb != b)]
        if len(nb) > EGO_CAP:
            weight = Aw[a, nb] + Aw[nb, a] + Aw[b, nb] + Aw[nb, b]
            nb = nb[np.argsort(-weight, kind="mergesort")[:EGO_CAP]]
        nodes = np.concatenate([[a, b], nb])
        S = (Aw[np.ix_(nodes, nodes)] > 0).astype(np.float64)
        np.fill_diagonal(S, 0)
        k = len(nodes)
        edges_k = S.sum()
        outdeg = S.sum(axis=1); indeg = S.sum(axis=0)
        br = float((outdeg >= 2).sum()); mg = float((indeg >= 2).sum())
        S2 = S @ S; S3 = S2 @ S; S4 = S3 @ S; S5 = S4 @ S
        routes = [S2[0, 1], S3[0, 1], S4[0, 1], S5[0, 1]]
        eidx = grp["e"].to_numpy()
        ego["irr_ego_nodes"][eidx] = k
        ego["irr_ego_edges"][eidx] = edges_k
        ego["irr_ego_density"][eidx] = edges_k / max(k * (k - 1), 1)
        ego["irr_branching_nodes"][eidx] = br
        ego["irr_merging_nodes"][eidx] = mg
        ego["irr_branch_merge_interleaving"][eidx] = min(br, mg) / max(max(br, mg), 1)
        ego["irr_routes_2to5"][eidx] = min(float(sum(routes)), 1e4)
        ego["irr_route_length_diversity"][eidx] = float(sum(1 for r in routes if r > 0))
        ego["irr_local_cycle_support"][eidx] = min(float(S2[0, 0] + S3[0, 0] + S2[1, 1] + S3[1, 1]), 1e4)
        ego["irr_asset_hop_diversity"][eidx] = float(assets_by_pair_day.get((dday, a), 1))
        ego["irr_complexity"][eidx] = np.log1p(br) + np.log1p(mg) + np.log1p(min(float(sum(routes)), 1e4)) * (1.0 if S2[1, 0] + S3[1, 0] > 0 else 0.5)
    out.update(ego)
    return pd.DataFrame(out, index=g.edges.index)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def motif_features(g: Graph, views: dict, prof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import time
    t0 = time.time()
    fm, pairs = forward_pairs(g)
    t1 = time.time()
    hw = hub_window_features(g, prof, fm)
    t2 = time.time()
    sg, wit_sg = scatter_gather(g, pairs, prof)
    t3 = time.time()
    cc, wit_cc = cycles_and_chains(g, pairs, prof)
    t4 = time.time()
    bi = bipartite_and_irregular(g, views, prof)
    t5 = time.time()
    hashes = g.edges["transaction_hash"].to_numpy()
    witnesses = {}
    wf, wb = fm["match_witness_fwd"].to_numpy(), fm["match_witness_bwd"].to_numpy()
    for e in range(g.n_edges):
        entry = {}
        if wf[e] >= 0:
            entry["forward_match"] = [hashes[e], hashes[wf[e]]]
        if wb[e] >= 0:
            entry["backward_match"] = [hashes[wb[e]], hashes[e]]
        if e in wit_sg:
            entry["scatter_gather"] = wit_sg[e]
        if e in wit_cc:
            entry[wit_cc[e]["motif"]] = wit_cc[e]
        if entry:
            witnesses[hashes[e]] = entry
    feats = pd.concat([fm.drop(columns=["match_witness_fwd", "match_witness_bwd"]), hw, sg, cc, bi], axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan).astype("float32")
    feats.attrs["timings"] = {"match": t1 - t0, "hub_windows": t2 - t1, "scatter_gather": t3 - t2,
                              "cycles_chains": t4 - t3, "bipartite_irregular": t5 - t4, "pairs": int(len(pairs))}
    print(f"    motif timings: {feats.attrs['timings']}", flush=True)
    return feats, witnesses
