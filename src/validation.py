"""Leakage-aware validation: unlabelled flow groups, blocked folds, metrics.

Primary protocol: StratifiedGroupKFold over *flow groups* built from
unlabelled graph structure only (value-matched forwards, same-wallet bursts,
repeated pairs; high-degree service hubs do not link). Labels are used for
stratification of fold assignment only, never as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import N_FOLDS, REVIEW_FRAC, SEED, VALUE_MATCH_SCALE
from src.graph_build import Graph


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def labels(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(len(self.p))])
        _, inv = np.unique(roots, return_inverse=True)
        return inv


def flow_groups(g: Graph, hub_quantile: float = 0.97, burst_gap: int = 600, forward_window: int = 6 * 3600,
                min_quality: float = 0.6, max_largest_frac: float = 0.15, community: np.ndarray | None = None
                ) -> tuple[np.ndarray, dict]:
    """Union-find groups of economic edges from unlabelled structure.

    If one giant group remains (dense 700-wallet graph), it is sub-split by the
    Louvain community of the sender, which is also label-free.
    """
    n = g.n_edges
    total_tx = np.bincount(g.src, minlength=g.n_wallets) + np.bincount(g.dst, minlength=g.n_wallets)
    info: dict = {}
    best_labels, best_info, best_frac = None, None, np.inf
    for hq in (hub_quantile, 0.95, 0.90, 0.80):
        cutoff = max(np.quantile(total_tx[total_tx > 0], hq), 50)
        is_hub = total_tx >= cutoff
        uf = UnionFind(n)
        # (a) same sender burst, (b) same receiver burst (non-hub)
        for wallet_codes, index in ((g.src, g.out_index), (g.dst, g.in_index)):
            w, t, eid = index.w, index.t, index.eid
            same = (w[1:] == w[:-1]) & ((t[1:] - t[:-1]) <= burst_gap) & (~is_hub[w[1:]])
            for i in np.where(same)[0]:
                uf.union(int(eid[i]), int(eid[i + 1]))
        # (c) value-matched forward through a non-hub intermediate
        oi = g.out_index
        lo, hi = oi.bounds(g.dst, g.ts, 0, forward_window)
        usd_in = np.nan_to_num(g.usd_recv)
        for e in range(n):
            b = g.dst[e]
            if is_hub[b] or hi[e] <= lo[e] or usd_in[e] <= 0:
                continue
            t_out, u_out, _cp, eids = oi.slice_arrays(int(lo[e]), int(hi[e]))
            later = (t_out > g.ts[e]) | ((t_out == g.ts[e]) & (eids > e))
            if not later.any():
                continue
            q = np.exp(-np.abs(np.log(np.maximum(u_out[later], 1e-9) / usd_in[e])) / VALUE_MATCH_SCALE)
            for k in np.where(q >= min_quality)[0]:
                uf.union(e, int(eids[later][k]))
        # (d) repeated pair within forward_window
        pairs = pd.DataFrame({"s": g.src, "d": g.dst, "t": g.ts, "e": np.arange(n)}).sort_values(["s", "d", "t"])
        same_pair = (pairs["s"].to_numpy()[1:] == pairs["s"].to_numpy()[:-1]) & (pairs["d"].to_numpy()[1:] == pairs["d"].to_numpy()[:-1])
        close = (pairs["t"].to_numpy()[1:] - pairs["t"].to_numpy()[:-1]) <= forward_window
        ee = pairs["e"].to_numpy()
        for i in np.where(same_pair & close)[0]:
            uf.union(int(ee[i]), int(ee[i + 1]))
        labels = uf.labels()
        sizes = np.bincount(labels)
        info = {"hub_quantile": hq, "n_hubs": int(is_hub.sum()), "hub_cutoff_tx": float(cutoff),
                "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
                "largest_frac": float(sizes.max() / n), "singleton_frac": float((sizes == 1).sum() / n)}
        if sizes.max() / n < best_frac:
            best_labels, best_info, best_frac = labels, dict(info), sizes.max() / n
        if sizes.max() / n <= max_largest_frac:
            break
    labels, info = best_labels, best_info
    sizes = np.bincount(labels)
    if sizes.max() / n > max_largest_frac and community is not None:
        giant = int(np.argmax(sizes))
        m = labels == giant
        sub = community[g.src[m]]
        labels = labels.copy()
        labels[m] = labels.max() + 1 + sub
        _, labels = np.unique(labels, return_inverse=True)
        sizes = np.bincount(labels)
        info.update({"giant_split_by_community": True, "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
                     "largest_frac": float(sizes.max() / n)})
    return labels, info


def make_folds(groups: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y, groups)]
    for tr, va in folds:
        if set(tr) & set(va):
            raise AssertionError("a row appears on both sides of a fold")
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("a flow group straddles train and validation")
    return folds


def time_holdout(ts: np.ndarray, holdout_days: float = 2.0, embargo_sec: int = 6 * 3600) -> tuple[np.ndarray, np.ndarray]:
    cut = ts.max() - holdout_days * 86400
    val = np.where(ts >= cut)[0]
    tr = np.where(ts < cut - embargo_sec)[0]
    return tr, val


def wallet_overlap(folds, g: Graph) -> list[float]:
    """Fraction of validation edges whose endpoints also appear in training edges (documented limit)."""
    out = []
    for tr, va in folds:
        seen = set(g.src[tr].tolist()) | set(g.dst[tr].tolist())
        both = np.array([(g.src[i] in seen) or (g.dst[i] in seen) for i in va])
        out.append(float(both.mean()))
    return out


def evaluate(y: np.ndarray, scores: np.ndarray, hashes: np.ndarray, difficulty: np.ndarray | None = None) -> dict:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("evaluation set has one class")
    ap = float(average_precision_score(y, scores))
    auc = float(roc_auc_score(y, scores))
    k = max(1, int(round(REVIEW_FRAC * len(y))))
    order = np.lexsort((hashes, -scores))[:k]
    top = np.zeros(len(y), dtype=bool)
    top[order] = True
    res = {"ap": ap, "ap100": 100 * ap, "roc_auc": auc, "n": int(len(y)), "positives": int(y.sum()),
           "precision_at_1pct": float(y[top].mean()), "recall_at_1pct": float(y[top].sum() / max(y.sum(), 1))}
    if difficulty is not None:
        d = pd.Series(difficulty).fillna("none").to_numpy()
        for lvl in sorted(set(d)):
            m = (d == lvl) & (y == 1)
            if m.sum():
                res[f"recall_at_1pct_{lvl}"] = float(top[m].mean())
                res[f"n_pos_{lvl}"] = int(m.sum())
    return res
