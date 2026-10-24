"""Graph A (economic payments) indexes used by every feature module.

* ``Graph`` holds integer wallet codes, ledger-ordered edges, and two sorted
  ``WindowIndex`` views (outgoing by sender, incoming by receiver) that answer
  "events of wallet w in [t-back, t+fwd]" for every edge at once with two
  ``searchsorted`` calls.
* ``static_views`` gives the collapsed undirected graph used for components,
  k-core, clustering and Louvain communities.

Wallet codes are split-local and never used as model features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import MAX_SLICE, SEED

KEY_MULT = np.int64(1) << np.int64(34)


class WindowIndex:
    """Events sorted by (wallet, time, ledger order) with vectorised range lookup."""

    def __init__(self, wallet: np.ndarray, times: np.ndarray, order_key: np.ndarray,
                 cparty: np.ndarray, usd: np.ndarray, edge_ids: np.ndarray):
        order = np.lexsort((order_key, times, wallet))
        self.order = order
        self.w = wallet[order].astype(np.int64)
        self.t = times[order].astype(np.int64)
        self.cp = cparty[order].astype(np.int64)
        self.usd = np.nan_to_num(usd[order].astype(np.float64))
        self.eid = edge_ids[order].astype(np.int64)
        self.key = self.w * KEY_MULT + self.t
        self.cum = np.concatenate([[0.0], np.cumsum(self.usd)])
        self.cum_sq = np.concatenate([[0.0], np.cumsum(self.usd ** 2)])
        self.cp_l = self.cp.tolist()
        self.usd_l = self.usd.tolist()
        self.t_l = self.t.tolist()
        self.eid_l = self.eid.tolist()
        # per-wallet contiguous block boundaries
        self.wallet_start = np.searchsorted(self.w, np.arange(int(wallet.max()) + 2 if len(wallet) else 1), side="left")

    def bounds(self, wq: np.ndarray, tq: np.ndarray, back: int, fwd: int):
        wq = wq.astype(np.int64)
        tq = tq.astype(np.int64)
        lo = np.searchsorted(self.key, wq * KEY_MULT + (tq - back), side="left")
        hi = np.searchsorted(self.key, wq * KEY_MULT + (tq + fwd), side="right")
        return lo, hi

    def strict_before(self, wq, tq):
        """Index of first event at time >= tq (used for as-of features)."""
        return np.searchsorted(self.key, wq.astype(np.int64) * KEY_MULT + tq.astype(np.int64), side="left")

    def strict_after(self, wq, tq):
        return np.searchsorted(self.key, wq.astype(np.int64) * KEY_MULT + tq.astype(np.int64), side="right")

    def counts_and_value(self, wq, tq, back, fwd):
        lo, hi = self.bounds(wq, tq, back, fwd)
        return (hi - lo).astype(np.float64), self.cum[hi] - self.cum[lo], lo, hi

    def stats(self, lo, hi):
        """count, sum, mean, std over slices (vectorised through cumulative sums)."""
        n = (hi - lo).astype(np.float64)
        s = self.cum[hi] - self.cum[lo]
        s2 = self.cum_sq[hi] - self.cum_sq[lo]
        mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        var = np.where(n > 0, s2 / np.maximum(n, 1) - mean ** 2, 0.0)
        return n, s, mean, np.sqrt(np.clip(var, 0, None))

    def distinct(self, lo, hi) -> np.ndarray:
        cp = self.cp_l
        out = np.empty(len(lo), dtype=np.float64)
        for i in range(len(lo)):
            a, b = int(lo[i]), int(hi[i])
            if b <= a:
                out[i] = 0.0
            elif b - a == 1:
                out[i] = 1.0
            else:
                out[i] = len(set(cp[a:min(b, a + MAX_SLICE)]))
        return out

    def max_usd(self, lo, hi) -> np.ndarray:
        usd = self.usd_l
        out = np.zeros(len(lo))
        for i in range(len(lo)):
            a, b = int(lo[i]), int(hi[i])
            if b > a:
                out[i] = max(usd[a:min(b, a + MAX_SLICE)])
        return out

    def best_amount_match(self, lo, hi, target: np.ndarray) -> np.ndarray:
        """Smallest relative USD gap in the slice. 1.0 = nothing comparable."""
        usd = self.usd_l
        out = np.ones(len(lo))
        for i in range(len(lo)):
            a, b = int(lo[i]), int(hi[i])
            tv = target[i]
            if b <= a or not np.isfinite(tv) or tv <= 0:
                continue
            best = 1.0
            for v in usd[a:min(b, a + MAX_SLICE)]:
                d = abs(v - tv) / tv
                if d < best:
                    best = d
                    if best < 1e-3:
                        break
            out[i] = min(best, 1.0)
        return out

    def nearest_gap(self, wq, tq, back, fwd, forward: bool) -> np.ndarray:
        lo, hi = self.bounds(wq, tq, back, fwd)
        out = np.full(len(lo), np.inf)
        idx = np.where(hi > lo)[0]
        if forward:
            out[idx] = np.abs(self.t[lo[idx]] - tq[idx])
        else:
            out[idx] = np.abs(tq[idx] - self.t[hi[idx] - 1])
        return out

    def slice_arrays(self, a: int, b: int):
        b = min(b, a + MAX_SLICE)
        return self.t[a:b], self.usd[a:b], self.cp[a:b], self.eid[a:b]


@dataclass
class Graph:
    edges: pd.DataFrame            # ledger-ordered economic edges
    wallets: pd.Index              # wallet address by code
    src: np.ndarray                # sender code per edge
    dst: np.ndarray                # receiver code per edge
    ts: np.ndarray                 # unix seconds per edge
    order_key: np.ndarray          # ledger order rank per edge (0..n-1)
    usd: np.ndarray                # paid USD per edge (NaN -> 0 in indexes)
    usd_recv: np.ndarray
    out_index: WindowIndex         # keyed by sender
    in_index: WindowIndex          # keyed by receiver

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_wallets(self) -> int:
        return len(self.wallets)


def build_graph(edges: pd.DataFrame) -> Graph:
    """Build Graph A indexes from the canonical economic-edge table."""
    e = edges.sort_values(["ledger_index", "transaction_index", "transaction_hash"], kind="mergesort").reset_index(drop=True)
    if e["transaction_hash"].duplicated().any():
        raise AssertionError("duplicate economic edge for one transaction hash")
    wallets = pd.Index(pd.unique(np.concatenate([e["source_wallet"].to_numpy(), e["destination_wallet"].to_numpy()])))
    wmap = {w: i for i, w in enumerate(wallets)}
    src = e["source_wallet"].map(wmap).to_numpy().astype(np.int64)
    dst = e["destination_wallet"].map(wmap).to_numpy().astype(np.int64)
    ts = e["timestamp_unix"].to_numpy().astype(np.int64)
    order_key = np.arange(len(e), dtype=np.int64)
    usd = e["paid_usd"].to_numpy(dtype=float)
    usd_recv = e["delivered_usd"].to_numpy(dtype=float)
    eid = np.arange(len(e), dtype=np.int64)
    out_index = WindowIndex(src, ts, order_key, dst, usd, eid)
    in_index = WindowIndex(dst, ts, order_key, src, usd_recv, eid)
    return Graph(e, wallets, src, dst, ts, order_key, usd, usd_recv, out_index, in_index)


def static_views(g: Graph) -> dict:
    """Collapsed undirected structure: components, k-core, clustering, Louvain, pagerank."""
    import networkx as nx

    n = g.n_wallets
    agg = pd.DataFrame({"s": g.src, "d": g.dst, "u": np.nan_to_num(g.usd)}).groupby(["s", "d"])["u"].agg(["sum", "size"]).reset_index()
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    D.add_weighted_edges_from(zip(agg["s"], agg["d"], agg["sum"].astype(float)))
    U = nx.Graph(D.to_undirected())
    comp = np.zeros(n, dtype=np.int64)
    for i, cset in enumerate(nx.connected_components(U)):
        for v in cset:
            comp[v] = i
    core = nx.core_number(U)
    clus = nx.clustering(U)
    try:
        pr = nx.pagerank(D, alpha=0.85, weight="weight", max_iter=100, tol=1e-6)
    except Exception:
        pr = {v: 1.0 / n for v in range(n)}
    try:
        comms = nx.community.louvain_communities(U, weight="weight", seed=SEED, resolution=1.0)
        comm = np.zeros(n, dtype=np.int64)
        for i, cset in enumerate(comms):
            for v in cset:
                comm[v] = i
    except Exception:
        comm = comp.copy()
    return {
        "component": comp,
        "core": np.array([core.get(v, 0) for v in range(n)], dtype=float),
        "clustering": np.array([clus.get(v, 0.0) for v in range(n)], dtype=float),
        "pagerank": np.array([pr.get(v, 0.0) for v in range(n)], dtype=float),
        "community": comm,
        "pair_counts": agg,
        "digraph": D,
    }
