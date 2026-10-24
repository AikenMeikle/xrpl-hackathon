"""Synthetic typology fixtures for the eight detectors (+ legitimate look-alikes)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.features_motifs import (bipartite_and_irregular, cycles_and_chains, forward_pairs, hub_window_features,
                                 motif_features, quality, scatter_gather)
from src.features_wallet import wallet_features
from src.graph_build import build_graph, static_views
from src.validation import flow_groups

T0 = 1_788_220_800
XRP = "0x1111222211112222111122221111222211112222:XRP"
USD = "rIssuerA:USD"


def make_edges(rows):
    """rows: (src, dst, t_offset_sec, usd[, asset_out])"""
    recs = []
    for i, r in enumerate(rows):
        s, d, dt, usd = r[:4]
        asset_out = r[4] if len(r) > 4 else XRP
        recs.append(dict(transaction_hash=f"H{i:04d}", source_wallet=s, destination_wallet=d, timestamp_unix=T0 + int(dt),
                         ledger_index=i + 1, transaction_index=0, paid_usd=float(usd), delivered_usd=float(usd) * 0.99,
                         paid_asset_id=XRP, delivered_asset_id=asset_out, conversion_flag=asset_out != XRP,
                         fee_drops=12, transfer_leg_count=1, in_economic_graph=True))
    return pd.DataFrame(recs)


def run(rows):
    edges = make_edges(rows)
    g = build_graph(edges)
    views = static_views(g)
    _wf, prof = wallet_features(g, views)
    feats, wit = motif_features(g, views, prof)
    feats = feats.copy()
    feats.index = g.edges["transaction_hash"].to_numpy()
    return feats, wit, g


def background(n=12, start=0, step=7200, usd=50):
    """Unrelated low-noise peer payments so profiles are not degenerate."""
    return [(f"bg{i}", f"bg{(i + 1) % n}", start + i * step, usd + i) for i in range(n)]


def test_quality_function():
    assert quality(100, 100) == pytest.approx(1.0)
    assert quality(50, 100) < quality(90, 100) < 1.0
    assert quality(100, 100) > quality(100, 1)


def test_fan_in_collector_gets_sweep_signal_merchant_does_not():
    deposits = [(f"s{i}", "collector", 60 * i, 100) for i in range(8)]
    sweep = [("collector", "exit", 60 * 8 + 300, 790)]
    # merchant: sustained receipts over days from repeat customers, no sweep
    merchant = [(f"c{i % 4}", "shop", 86400 * (i // 4) + 3600 * (i % 4), 20 + i) for i in range(16)]
    feats, wit, g = run(deposits + sweep + merchant + background())
    dep = feats.loc["H0003"]
    swp = feats.loc["H0008"]
    assert dep["fanin_in_degree_6h"] >= 4
    assert dep["fanin_sweep_quality"] > 0.5  # exit matches the collected total
    assert dep["fanin_collector_exit_usd_1h"] > 700
    assert swp["fanin_exit_support_senders_6h"] == 8
    assert swp["fanin_exit_sweep_quality"] > 0.8
    shop = feats.loc["H0020"]  # a merchant receipt
    assert shop["fanin_sweep_quality"] == 0 and shop["fanin_collector_exit_usd_1h"] == 0
    assert shop["fanin_one_use_sender_frac"] == 0


def test_fan_out_uniform_payouts_forwarding_and_funding_are_separate():
    funding = [("funder", "src", 0, 1000)]
    payouts = [("src", f"r{i}", 300 + 30 * i, 100) for i in range(10)]
    forwards = [(f"r{i}", f"x{i}", 600 + 30 * i, 99) for i in range(5)]  # half the recipients forward
    feats, wit, g = run(funding + payouts + forwards + background())
    p = feats.loc["H0005"]
    assert p["fanout_payout_distinct_1h"] == 10
    assert p["fanout_payout_cv_1h"] < 0.05
    assert p["fanout_equal_size_frac_1h"] == 1.0
    assert p["fanout_just_funded_ratio_1h"] == pytest.approx(1.0, rel=0.05)
    assert 0.4 <= p["fanout_recipient_forward_frac_1h"] <= 0.6
    f0 = feats.loc["H0000"]  # funding edge projected: receiver splits within 1h
    assert f0["fanout_funds_split_deg_1h"] == 10
    # payroll look-alike: same uniform payouts but recipients keep the money and the source was not just funded
    payroll = [("employer", f"emp{i}", 86400 * 3 + 30 * i, 100) for i in range(10)]
    feats2, _, _ = run(payroll + background())
    q = feats2.loc["H0005"]
    assert q["fanout_payout_cv_1h"] < 0.05 and q["fanout_recipient_forward_frac_1h"] == 0 and q["fanout_just_funded_ratio_1h"] == 0


def test_scatter_gather_distinct_middlemen_in_time_order():
    scatter = [("S", f"m{i}", 0 + 10 * i, 200) for i in range(4)]
    gather = [(f"m{i}", "D", 3600 + 10 * i, 195) for i in range(4)]
    wrong_order = [("m9", "D", 0, 195), ("S", "m9", 7200, 200)]  # gather before scatter: not a branch
    feats, wit, g = run(scatter + gather + wrong_order + background())
    assert feats.loc["H0000", "sg_middlemen_count"] == 4
    assert feats.loc["H0005", "sg_middlemen_count"] == 4
    assert feats.loc["H0000", "sg_value_match_quality"] > 0.8
    assert feats.loc["H0008", "sg_middlemen_count"] == 0 or feats.loc["H0008", "sg_middlemen_count"] <= 4
    assert "scatter_gather" in wit["H0000"] and len(wit["H0000"]["scatter_gather"]) >= 8


def test_gather_scatter_disjoint_counterparties_and_balance():
    ins = [(f"a{i}", "hub", 60 * i, 100) for i in range(6)]
    outs = [("hub", f"b{i}", 1800 + 60 * i, 98) for i in range(6)]
    feats, wit, g = run(ins + outs + background())
    e = feats.loc["H0002"]
    assert e["gs_dst_in_deg_6h"] == 6 and e["gs_dst_out_deg_6h"] == 6
    assert e["gs_dst_disjointness_6h"] == 1.0
    assert e["gs_dst_pass_through_6h"] > 0.95 and e["gs_dst_flow_mismatch_6h"] < 0.05
    assert e["gs_dst_exits_after_funding_frac_1h"] == 1.0


def test_cycle_counted_only_when_chronological():
    good = [("A", "B", 0, 500), ("B", "C", 600, 495), ("C", "A", 1200, 490)]
    feats, wit, g = run(good + background())
    assert feats.loc["H0000", "cycle_count_total"] >= 1
    assert feats.loc["H0000", "cycle_shortest_return_hops"] == 3
    assert feats.loc["H0001", "cycle_edge_participation"] >= 1
    assert wit["H0000"]["cycle"]["hashes"] == ["H0000", "H0001", "H0002"]
    reversed_times = [("A", "B", 1200, 500), ("B", "C", 600, 495), ("C", "A", 0, 490)]
    feats2, _, _ = run(reversed_times + background())
    assert feats2["cycle_count_total"].sum() == 0
    # dust-only bot cycle: tiny economic scale yields no cycle evidence
    dust = [("A", "B", 0, 0.1), ("B", "A", 60, 0.1)]
    feats3, _, _ = run(dust + background())
    assert feats3.loc["H0000", "cycle_count_total"] == 0


def test_layered_stack_longest_path_and_single_use_intermediaries():
    chain = [("w0", "w1", 0, 1000), ("w1", "w2", 900, 980), ("w2", "w3", 1800, 960), ("w3", "w4", 2700, 940), ("w4", "w5", 3600, 920)]
    feats, wit, g = run(chain + background())
    assert feats.loc["H0002", "chain_longest_hops"] == 5
    assert feats.loc["H0002", "chain_single_use_intermediary_frac"] == 1.0
    assert feats.loc["H0002", "chain_hop_retention_median"] > 0.8
    assert feats.loc["H0002", "chain_peel_ratio_median"] < 1.0
    assert wit["H0002"]["chain"]["hashes"] == ["H0000", "H0001", "H0002", "H0003", "H0004"]


def test_bipartite_rectangle_without_directed_cycle():
    block = [(a, b, 100 * i, 50 + i) for i, (a, b) in enumerate([("p1", "q1"), ("p1", "q2"), ("p2", "q1"), ("p2", "q2"), ("p1", "q3"), ("p2", "q3")])]
    feats, wit, g = run(block + background())
    assert feats.loc["H0000", "bip_rect_count_full"] >= 2
    assert feats.loc["H0000", "bip_block_payers"] == 2 and feats.loc["H0000", "bip_block_receivers"] == 3
    assert feats.loc["H0000", "bip_block_density"] == 1.0
    assert feats["cycle_count_total"].sum() == 0  # no directed money return


def test_irregular_network_bounded_runtime():
    rng = np.random.default_rng(0)
    rows = [("o", "x1", 0, 900), ("o", "x2", 100, 600), ("x1", "y", 500, 450), ("x1", "z", 600, 440),
            ("x2", "y", 700, 590), ("y", "o", 1500, 800), ("z", "q", 1600, 430), ("y", "q", 1700, 200), ("q", "o", 3000, 600)]
    rows += [(f"n{rng.integers(0, 40)}", f"n{rng.integers(0, 40)}", int(rng.integers(0, 6 * 86400)), float(rng.uniform(5, 500))) for _ in range(400)]
    rows = [r for r in rows if r[0] != r[1]]
    t = time.time()
    feats, wit, g = run(rows + background())
    assert time.time() - t < 60
    e = feats.loc["H0000"]
    assert e["irr_branching_nodes"] >= 2 and e["irr_merging_nodes"] >= 2
    assert e["irr_routes_2to5"] >= 2
    assert e["irr_local_cycle_support"] >= 1
    assert feats[["irr_ego_nodes", "irr_ego_density", "irr_complexity"]].notna().all().all()


def test_three_leg_conversion_is_one_edge_in_graph():
    # one economic edge (A->B) with a conversion flag: never three hops A->LP->LP->B
    rows = [("A", "B", 0, 100, USD)] + background()
    feats, wit, g = run(rows)
    assert g.n_edges == len(rows)
    assert "LP" not in "".join(g.wallets.tolist())
    assert feats.loc["H0000", "chain_longest_hops"] <= 2


def test_flow_groups_do_not_use_labels_and_cover_all_edges():
    rows = [("S", f"m{i}", 10 * i, 200) for i in range(4)] + [(f"m{i}", "D", 3600 + 10 * i, 195) for i in range(4)] + background()
    g = build_graph(make_edges(rows))
    groups, info = flow_groups(g)
    assert len(groups) == g.n_edges
    assert len(set(groups[:8])) == 1  # the whole scatter-gather is one group
