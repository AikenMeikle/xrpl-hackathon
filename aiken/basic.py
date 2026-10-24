#!/usr/bin/env python3
"""
aml_detector.py -- UKCBT "Decode & Detect" end-to-end suspicious-payment detector.

    python aml_detector.py --train train --test test-data --out predictions.csv

Design in five steps:

  1. LOAD + NORMALISE. Parquet (or CSV) -> one "economic edge" per transaction:
     first transfer leg = what the sender actually paid, last leg = what the
     receiver actually got, both valued in USD off market_rates.csv.

  2. FEATURES (~140, all structural). Wallet addresses are disjoint between
     train and test, so no identity feature can ever be used -- only shape:
     burst windows, value conservation, two-hop funnels, graph centrality,
     community structure, amount regularity.

  3. STAGE 1. Gradient boosting, cross-validated with folds grouped by graph
     component, so a planted scenario can never straddle train and validation.
     That makes the out-of-fold scores honest.

  4. STAGE 2 -- "guilt by association". Laundering is never one transaction.
     Feed each row the stage-1 scores of its *neighbours* (same sender, same
     receiver, downstream wallet, upstream wallet, same community) with self
     excluded, then retrain. This is the single biggest lift in the pipeline
     and it is purely transductive -- it reads predictions, never test labels.

  5. WRITE. One row per test transaction, exactly once, score in [0, 1].

No silent fallbacks: if the model cannot be fit, or held-out AP collapses to
the base rate, the script fails loudly rather than emitting noise.

Deps: pandas, numpy, pyarrow, scikit-learn.  Optional: lightgbm, networkx.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG -- every tunable in one place
# ----------------------------------------------------------------------------

# Sliding windows for burst / counterparty-count features (seconds).
WINDOWS = {"10m": 600, "1h": 3600, "6h": 21600, "24h": 86400}

# Follow-on windows for two-stage patterns (gather-scatter, scatter-gather).
FOLLOW_WINDOWS = {"1h": 3600, "6h": 21600, "24h": 86400}

# Two amounts count as "the same money moving on" within this relative gap.
PASSTHROUGH_TOL = 0.10

# Cap on slice length in per-row reductions (bounds worst-case hub cost).
MAX_SLICE = 4000

# XRP sentinel token address from the data dictionary.
XRP_SENTINEL = "0x1111222211112222111122221111222211112222"

# Cross-validation.
N_FOLDS = 5
SEED = 20261024

# Review budget used for the precision/recall readout (site uses top 1%).
REVIEW_FRAC = 0.01

KEY_MULT = np.int64(1) << np.int64(32)  # wallet/time composite sort key

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------

def _find(dirpath: Path, stem: str) -> Path | None:
    """Locate <stem>.parquet/.csv in dir, or one level down (handles the
    'unzipped into a wrapper folder' case that bites everyone at 2am)."""
    cands = []
    for base in [dirpath, *sorted(p for p in dirpath.iterdir() if p.is_dir())]:
        for ext in (".parquet", ".csv"):
            cands.append(base / f"{stem}{ext}")
    for c in cands:
        if c.exists():
            return c
    return None


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_split(data_dir: str | Path) -> dict:
    """Load one split. blocks.parquet is deliberately NOT used: its
    TRANSACTION_HASH is not a per-transaction foreign key (see the brief), and
    nothing in these topologies needs ledger-level metadata."""
    d = Path(data_dir).expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")

    tx_p, tf_p, mr_p = _find(d, "transactions"), _find(d, "transfers"), _find(d, "market_rates")
    if tx_p is None or tf_p is None:
        raise FileNotFoundError(
            f"{d} has no transactions/transfers file. Point --train at the folder "
            f"holding transactions.parquet (e.g. the 'train' folder itself)."
        )
    tx, tf = _read(tx_p), _read(tf_p)
    rates = _read(mr_p) if mr_p is not None else pd.DataFrame()

    lab = None
    for cand in (d / "labels.csv", d.parent / "labels.csv", d / "train" / "labels.csv"):
        if cand.exists():
            lab = pd.read_csv(cand)
            break

    log(f"{d.name}: {len(tx):,} transactions, {len(tf):,} legs, {len(rates):,} price rows"
        + (f", {len(lab):,} labels" if lab is not None else ", no labels"))
    return {"tx": tx, "tf": tf, "rates": rates, "labels": lab, "dir": d}


# ----------------------------------------------------------------------------
# 2. ECONOMIC EDGE + USD VALUATION
# ----------------------------------------------------------------------------

def asset_key(token_address: pd.Series, symbol: pd.Series) -> pd.Series:
    """Asset identity = (issuer, currency). USD from issuer X != USD from Y."""
    sym = symbol.astype(str)
    tok = token_address.astype(str)
    is_xrp = (sym.str.upper() == "XRP") | (tok.str.lower() == XRP_SENTINEL.lower())
    return np.where(is_xrp, "XRP", tok + ":" + sym)


def price_legs_usd(tf: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Attach usd_value to every transfer leg using the hourly reference price.

    merge_asof requires the `on` key to be monotonic across the WHOLE frame --
    not merely within each `by` group -- so both sides are sorted by hour alone.
    Backward direction means a sparse IOU series falls back to its last print
    instead of going NaN.
    """
    tf = tf.copy()
    tf["asset_key"] = asset_key(tf["TOKEN_ADDRESS"], tf["SYMBOL"])
    tf["hour_ts"] = (tf["TIMESTAMP"].astype("int64") // 3600) * 3600

    if len(rates) == 0:
        tf["usd_value"] = np.nan
        tf["price_usd"] = np.nan
        return tf

    r = rates.copy()
    r["asset_key"] = r["ASSET"].astype(str)
    r["hour_ts"] = r["TIMESTAMP"].astype("int64")
    r["price_usd"] = r["PRICE_MICRO_USD"].astype("float64") / 1e6
    r = r[["asset_key", "hour_ts", "price_usd"]].sort_values("hour_ts", kind="mergesort")

    left = tf[["asset_key", "hour_ts"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("hour_ts", kind="mergesort")

    merged = pd.merge_asof(left, r, on="hour_ts", by="asset_key", direction="backward")
    # Some legs sit before the first print for their asset -> try forward too.
    if merged["price_usd"].isna().any():
        fwd = pd.merge_asof(left, r, on="hour_ts", by="asset_key", direction="forward")
        merged["price_usd"] = merged["price_usd"].fillna(fwd["price_usd"])

    price = np.full(len(tf), np.nan)
    price[merged["_row"].to_numpy()] = merged["price_usd"].to_numpy()
    tf["price_usd"] = price
    tf["usd_value"] = tf["VALUE"].astype("float64") * tf["price_usd"]

    miss = float(np.mean(~np.isfinite(tf["usd_value"].to_numpy())))
    if miss > 0.02:
        log(f"  WARNING: {miss:.1%} of legs could not be priced in USD")
    return tf


def _parse_amount(obj) -> tuple[float, str]:
    """EXTRA_TRANSACTION_FIELDS amounts: drop-string for XRP, dict for IOU."""
    if obj is None:
        return (np.nan, "")
    if isinstance(obj, dict):
        try:
            return (float(obj.get("value", np.nan)), str(obj.get("currency", "")))
        except (TypeError, ValueError):
            return (np.nan, "")
    s = str(obj)
    try:
        return (float(s) / 1e6, "XRP")
    except ValueError:
        return (np.nan, "")


def build_edges(tx: pd.DataFrame, tf: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """One row per TRANSACTION_HASH: the economic sender->receiver edge.

    Guarantees full coverage of transactions.parquet -- a transaction whose legs
    are missing still gets a row, because the submission rejects missing IDs.
    """
    tf = price_legs_usd(tf, rates)
    tf = tf.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"], kind="mergesort")
    g = tf.groupby("TRANSACTION_HASH", sort=False)

    first = g.head(1).set_index("TRANSACTION_HASH")
    last = g.tail(1).set_index("TRANSACTION_HASH")
    n_legs = g.size().rename("n_legs")
    any_path = g["TRANSFER_TYPE"].apply(lambda s: bool((s == "path_payment").any())).rename("is_conversion")

    e = pd.DataFrame(index=first.index)
    e["sender"] = first["FROM_ADDRESS"].astype(str)          # real debit, not SendMax
    e["receiver"] = last["TO_ADDRESS"].astype(str)           # real credit
    e["TIMESTAMP"] = first["TIMESTAMP"].astype("int64")
    e["usd_sent"] = first["usd_value"].astype("float64")
    e["usd_recv"] = last["usd_value"].astype("float64")
    e["sent_symbol"] = first["SYMBOL"].astype(str)
    e["recv_symbol"] = last["SYMBOL"].astype(str)
    e["sent_asset"] = first["asset_key"].astype(str)
    e["recv_asset"] = last["asset_key"].astype(str)
    e["raw_sent"] = first["VALUE"].astype("float64")
    e = e.join(n_legs).join(any_path)
    e = e.reset_index()

    # Left-join onto the authoritative transaction list.
    keep = ["TRANSACTION_HASH", "TIMESTAMP", "ACCOUNT", "FEE", "TRANSACTION_TYPE",
            "EXTRA_TRANSACTION_FIELDS", "LEDGER_INDEX", "SEQUENCE"]
    t = tx[[c for c in keep if c in tx.columns]].copy()
    df = t.merge(e.drop(columns=["TIMESTAMP"]), on="TRANSACTION_HASH", how="left")

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    df["sender"] = df["sender"].fillna(df.get("ACCOUNT", pd.Series(index=df.index, dtype=object)))
    df["sender"] = df["sender"].fillna("__unknown_sender__").astype(str)
    df["receiver"] = df["receiver"].fillna("__unknown_receiver__").astype(str)
    df["n_legs"] = df["n_legs"].fillna(1).astype(int)
    df["is_conversion"] = df["is_conversion"].fillna(False).astype(bool)

    # Nominal destination / spending ceiling from the denormalised JSON blob.
    dest, sendmax, delivermax = [], [], []
    raw = df.get("EXTRA_TRANSACTION_FIELDS", pd.Series([None] * len(df)))
    for v in raw:
        d = None
        if isinstance(v, dict):
            d = v
        elif isinstance(v, str) and v.strip().startswith("{"):
            try:
                d = json.loads(v)
            except (ValueError, TypeError):
                d = None
        if not isinstance(d, dict):
            dest.append(None); sendmax.append(np.nan); delivermax.append(np.nan); continue
        dest.append(d.get("Destination"))
        sendmax.append(_parse_amount(d.get("SendMax"))[0])
        delivermax.append(_parse_amount(d.get("DeliverMax") or d.get("Amount"))[0])
    df["nominal_destination"] = dest
    df["sendmax_units"] = sendmax
    df["delivermax_units"] = delivermax

    df["usd_sent"] = df["usd_sent"].astype("float64")
    df["usd_recv"] = df["usd_recv"].astype("float64")
    med = np.nanmedian(df["usd_sent"].to_numpy()) if np.isfinite(df["usd_sent"]).any() else 1.0
    df["usd_sent"] = df["usd_sent"].fillna(med)
    df["usd_recv"] = df["usd_recv"].fillna(df["usd_sent"])
    df["TIMESTAMP"] = df["TIMESTAMP"].fillna(df["TIMESTAMP"].median()).astype("int64")

    df = df.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 3. SLIDING-WINDOW ENGINE
# ----------------------------------------------------------------------------

class WindowIndex:
    """Time-sorted per-wallet event blocks with O(1) vectorised range lookup.

    All events are sorted by a single composite int64 key (wallet << 32 | time),
    so the bounds of "wallet w between t-back and t+fwd" for ALL rows at once is
    two numpy searchsorted calls -- no Python loop over wallet pairs, no O(n^2).
    """

    def __init__(self, wallet: np.ndarray, times: np.ndarray,
                 cparty: np.ndarray, usd: np.ndarray, aux: np.ndarray | None = None):
        order = np.lexsort((times, wallet))
        self.w = wallet[order].astype(np.int64)
        self.t = times[order].astype(np.int64)
        self.cp = cparty[order].astype(np.int64)
        self.usd = np.nan_to_num(usd[order].astype(np.float64))
        self.aux = np.nan_to_num(aux[order].astype(np.float64)) if aux is not None else None
        self.key = self.w * KEY_MULT + self.t
        self.cum = np.concatenate([[0.0], np.cumsum(self.usd)])
        self.cp_l = self.cp.tolist()
        self.usd_l = self.usd.tolist()
        self.t_l = self.t.tolist()
        self.aux_l = self.aux.tolist() if self.aux is not None else None
        self.order = order

    def bounds(self, wq: np.ndarray, tq: np.ndarray, back: int, fwd: int):
        wq = wq.astype(np.int64)
        tq = tq.astype(np.int64)
        lo = np.searchsorted(self.key, wq * KEY_MULT + (tq - back), side="left")
        hi = np.searchsorted(self.key, wq * KEY_MULT + (tq + fwd), side="right")
        return lo, hi

    def counts_and_value(self, wq, tq, back, fwd):
        lo, hi = self.bounds(wq, tq, back, fwd)
        return (hi - lo).astype(np.float64), self.cum[hi] - self.cum[lo], lo, hi

    def distinct(self, lo, hi) -> np.ndarray:
        """Distinct counterparties per row. Python sets on list slices beat
        np.unique by a wide margin at these slice sizes."""
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

    def max_aux(self, lo, hi) -> np.ndarray:
        aux = self.aux_l
        out = np.zeros(len(lo), dtype=np.float64)
        if aux is None:
            return out
        for i in range(len(lo)):
            a, b = int(lo[i]), int(hi[i])
            if b > a:
                out[i] = max(aux[a:min(b, a + MAX_SLICE)])
        return out

    def best_amount_match(self, lo, hi, target: np.ndarray) -> np.ndarray:
        """Smallest relative amount gap in the slice -- 'is this same money
        moving on?'. 1.0 means nothing comparable found."""
        usd = self.usd_l
        out = np.ones(len(lo), dtype=np.float64)
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
        """Seconds to the nearest event in the window (inf if none)."""
        lo, hi = self.bounds(wq, tq, back, fwd)
        t = self.t
        out = np.full(len(lo), np.inf)
        has = hi > lo
        idx = np.where(has)[0]
        if forward:
            out[idx] = np.abs(t[lo[idx]] - tq[idx])
        else:
            out[idx] = np.abs(tq[idx] - t[hi[idx] - 1])
        return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Conservation of value: min/max, 1.0 = everything in came straight out."""
    a = np.nan_to_num(np.asarray(a, dtype=float))
    b = np.nan_to_num(np.asarray(b, dtype=float))
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.where(hi > 0, lo / np.maximum(hi, 1e-9), 0.0)


# ----------------------------------------------------------------------------
# 4. FEATURE BLOCKS
# ----------------------------------------------------------------------------

def f_intrinsic(df: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction shape. Cheap, weak alone, useful as GBM interactions."""
    f = pd.DataFrame(index=df.index)
    v = df["usd_sent"].to_numpy()
    f["usd_sent"] = v
    f["log_usd"] = np.log1p(np.clip(v, 0, None))
    f["usd_recv"] = df["usd_recv"].to_numpy()
    f["value_leak"] = ratio(df["usd_sent"], df["usd_recv"])      # conversion spread
    f["n_legs"] = df["n_legs"].to_numpy()
    f["is_conversion"] = df["is_conversion"].astype(int).to_numpy()
    f["cross_currency"] = (df["sent_symbol"] != df["recv_symbol"]).astype(int).to_numpy()
    f["self_payment"] = (df["sender"] == df["receiver"]).astype(int).to_numpy()

    fee = pd.to_numeric(df.get("FEE", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()
    f["fee_xrp"] = fee
    f["fee_ratio"] = np.where(v > 0, fee / np.maximum(v, 1e-9), 0.0)

    # Round-number structuring: planted amounts cluster on tidy figures.
    raw = np.nan_to_num(df["raw_sent"].to_numpy())
    f["is_round_10"] = (np.abs(raw - np.round(raw, -1)) < 1e-6).astype(int)
    f["is_round_100"] = (np.abs(raw - np.round(raw, -2)) < 1e-6).astype(int)
    f["is_round_1000"] = (np.abs(raw - np.round(raw, -3)) < 1e-6).astype(int)
    f["has_cents"] = (np.abs(raw - np.round(raw)) > 1e-6).astype(int)

    ts = df["TIMESTAMP"].to_numpy()
    f["hour_of_day"] = (ts // 3600) % 24
    f["day_index"] = (ts - ts.min()) // 86400
    f["is_night"] = ((f["hour_of_day"] < 6) | (f["hour_of_day"] >= 22)).astype(int)

    # SendMax headroom: how close to the ceiling did the payment actually run?
    sm = df["sendmax_units"].to_numpy()
    rs = df["raw_sent"].to_numpy()
    f["has_sendmax"] = np.isfinite(sm).astype(int)
    f["sendmax_headroom"] = np.where(np.isfinite(sm) & (sm > 0), rs / sm, np.nan)

    nom = df["nominal_destination"].astype(object)
    f["dest_mismatch"] = (nom.notna() & (nom.astype(str) != df["receiver"])).astype(int).to_numpy()
    return f


def f_wallet_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime behaviour of each wallet within the split. Structural only --
    never the address itself, since wallets are disjoint across splits."""
    out_g = df.groupby("sender")
    in_g = df.groupby("receiver")

    prof_out = pd.DataFrame({
        "o_tx": out_g.size(),
        "o_usd": out_g["usd_sent"].sum(),
        "o_deg": out_g["receiver"].nunique(),
        "o_mean": out_g["usd_sent"].mean(),
        "o_std": out_g["usd_sent"].std(),
        "o_first": out_g["TIMESTAMP"].min(),
        "o_last": out_g["TIMESTAMP"].max(),
        "o_conv": out_g["is_conversion"].mean(),
    })
    prof_in = pd.DataFrame({
        "i_tx": in_g.size(),
        "i_usd": in_g["usd_recv"].sum(),
        "i_deg": in_g["sender"].nunique(),
        "i_mean": in_g["usd_recv"].mean(),
        "i_std": in_g["usd_recv"].std(),
        "i_first": in_g["TIMESTAMP"].min(),
        "i_last": in_g["TIMESTAMP"].max(),
    })
    prof = prof_out.join(prof_in, how="outer").fillna(0.0)

    prof["life"] = np.maximum(prof[["o_last", "i_last"]].max(axis=1)
                              - prof[["o_first", "i_first"]].replace(0, np.nan).min(axis=1), 0).fillna(0)
    prof["throughput"] = ratio(prof["i_usd"], prof["o_usd"])      # pass-through wallet?
    prof["net_flow"] = prof["i_usd"] - prof["o_usd"]
    prof["deg_ratio"] = ratio(prof["i_deg"], prof["o_deg"])
    prof["o_cv"] = prof["o_std"] / np.maximum(prof["o_mean"], 1e-9)   # low CV = uniform splits
    prof["i_cv"] = prof["i_std"] / np.maximum(prof["i_mean"], 1e-9)
    prof["burstiness"] = (prof["o_tx"] + prof["i_tx"]) / np.maximum(prof["life"] / 3600.0, 1.0)

    cols = ["o_tx", "o_usd", "o_deg", "o_mean", "o_cv", "o_conv", "i_tx", "i_usd",
            "i_deg", "i_mean", "i_cv", "life", "throughput", "net_flow",
            "deg_ratio", "burstiness"]

    f = pd.DataFrame(index=df.index)
    for c in cols:
        f[f"snd_{c}"] = prof[c].reindex(df["sender"]).to_numpy()
        f[f"rcv_{c}"] = prof[c].reindex(df["receiver"]).to_numpy()

    # Wallet-pair history: is this a recurring commercial relationship, or new?
    pair = df.groupby(["sender", "receiver"])
    pc = pair.size().rename("pair_tx")
    pv = pair["usd_sent"].sum().rename("pair_usd")
    pr = df.set_index(["sender", "receiver"]).index
    f["pair_tx"] = pc.reindex(pr).to_numpy()
    f["pair_usd"] = pv.reindex(pr).to_numpy()
    f["pair_share_of_sender"] = f["pair_usd"] / np.maximum(f["snd_o_usd"], 1e-9)
    f["sender_edge_concentration"] = f["pair_tx"] / np.maximum(f["snd_o_tx"], 1)
    return f


def f_windows(df: pd.DataFrame, idx_out: WindowIndex, idx_in: WindowIndex,
              s_code: np.ndarray, r_code: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Fan-in / fan-out counts in every window, looking BOTH ways.

    Both directions matter: a transaction at the leading edge of a fan-out burst
    has no backward signal at all, and a pure backward view would miss it.
    """
    f = pd.DataFrame(index=df.index)
    for name, w in WINDOWS.items():
        for direction, (back, fwd) in {"b": (w, 0), "f": (0, w), "c": (w, w)}.items():
            # fan-out: distinct receivers this sender pays
            n, usd, lo, hi = idx_out.counts_and_value(s_code, ts, back, fwd)
            f[f"fanout_{name}_{direction}_n"] = n
            f[f"fanout_{name}_{direction}_usd"] = usd
            f[f"fanout_{name}_{direction}_deg"] = idx_out.distinct(lo, hi)

            # fan-in: distinct senders paying this receiver
            n, usd, lo, hi = idx_in.counts_and_value(r_code, ts, back, fwd)
            f[f"fanin_{name}_{direction}_n"] = n
            f[f"fanin_{name}_{direction}_usd"] = usd
            f[f"fanin_{name}_{direction}_deg"] = idx_in.distinct(lo, hi)

        # Concentration: a true fan-in burst is many DISTINCT wallets, not one
        # wallet paying repeatedly (that is a subscription, not a smurf ring).
        f[f"fanin_{name}_distinctness"] = (f[f"fanin_{name}_c_deg"]
                                           / np.maximum(f[f"fanin_{name}_c_n"], 1))
        f[f"fanout_{name}_distinctness"] = (f[f"fanout_{name}_c_deg"]
                                            / np.maximum(f[f"fanout_{name}_c_n"], 1))
    return f


def f_passthrough(df: pd.DataFrame, idx_out: WindowIndex, idx_in: WindowIndex,
                  s_code: np.ndarray, r_code: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Gather-scatter and scatter-gather cores: does value that arrives leave
    again, intact, soon, to different people?"""
    f = pd.DataFrame(index=df.index)
    usd = df["usd_sent"].to_numpy()

    for name, w in FOLLOW_WINDOWS.items():
        # --- GATHER-SCATTER: receiver collected, then disperses -------------
        in_n, in_usd, in_lo, in_hi = idx_in.counts_and_value(r_code, ts, w, 0)
        out_n, out_usd, out_lo, out_hi = idx_out.counts_and_value(r_code, ts, 0, w)
        in_deg = idx_in.distinct(in_lo, in_hi)
        out_deg = idx_out.distinct(out_lo, out_hi)

        f[f"gs_{name}_in_deg"] = in_deg
        f[f"gs_{name}_out_deg"] = out_deg
        f[f"gs_{name}_conservation"] = ratio(in_usd, out_usd)
        f[f"gs_{name}_forward_frac"] = out_usd / np.maximum(in_usd, 1e-9)
        f[f"gs_{name}_both"] = np.sqrt(np.clip(in_deg, 0, None) * np.clip(out_deg, 0, None))
        # Product form: only fires when collection AND dispersal AND value match.
        f[f"gs_{name}_score"] = (f[f"gs_{name}_both"] * f[f"gs_{name}_conservation"])
        f[f"gs_{name}_amount_echo"] = 1.0 - idx_out.best_amount_match(out_lo, out_hi, usd)

        # --- SCATTER-GATHER: sender dispersed, money reconverges ------------
        so_n, so_usd, so_lo, so_hi = idx_out.counts_and_value(s_code, ts, w, w)
        si_n, si_usd, si_lo, si_hi = idx_in.counts_and_value(s_code, ts, w, 0)
        f[f"sg_{name}_scatter_deg"] = idx_out.distinct(so_lo, so_hi)
        f[f"sg_{name}_sender_funded"] = ratio(si_usd, so_usd)
        f[f"sg_{name}_sender_in_deg"] = idx_in.distinct(si_lo, si_hi)

        # Two-hop funnel: of the wallets this receiver pays next, does any of
        # them sit at the mouth of its own fan-in? That is the gather leg.
        f[f"hop2_funnel_{name}"] = idx_out.max_aux(out_lo, out_hi)

        f[f"sg_{name}_score"] = f[f"sg_{name}_scatter_deg"] * f[f"hop2_funnel_{name}"]

    # Timing: hot money is forwarded in minutes, not days.
    f["secs_to_forward"] = np.minimum(
        idx_out.nearest_gap(r_code, ts, 0, 86400, forward=True), 86400)
    f["secs_since_funded"] = np.minimum(
        idx_in.nearest_gap(s_code, ts, 86400, 0, forward=False), 86400)
    f["log_to_forward"] = np.log1p(f["secs_to_forward"])
    f["log_since_funded"] = np.log1p(f["secs_since_funded"])
    f["u_turn"] = ((f["secs_to_forward"] < 3600) & (f["secs_since_funded"] < 3600)).astype(int)
    return f


def f_graph(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Centrality + community structure. Planted motifs are dense little
    subgraphs; ordinary commerce is a sprawling low-density mesh.

    Returns (features, cv_group_ids). Group ids keep a whole scenario inside a
    single CV fold, which is what makes the held-out AP trustworthy.
    """
    f = pd.DataFrame(index=df.index)
    senders = df["sender"].to_numpy()
    receivers = df["receiver"].to_numpy()
    nodes = pd.Index(pd.unique(np.concatenate([senders, receivers])))
    nmap = {a: i for i, a in enumerate(nodes)}
    su = np.array([nmap[a] for a in senders])
    ru = np.array([nmap[a] for a in receivers])

    # Union-find components (no dependency, linear time).
    parent = np.arange(len(nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(su, ru):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = np.array([find(i) for i in range(len(nodes))])
    _, comp = np.unique(comp, return_inverse=True)

    comp_nodes = np.bincount(comp)
    tx_comp = comp[su]
    comp_tx = np.bincount(tx_comp, minlength=comp_nodes.size)
    comp_usd = np.bincount(tx_comp, weights=np.nan_to_num(df["usd_sent"].to_numpy()),
                           minlength=comp_nodes.size)

    f["comp_nodes"] = comp_nodes[tx_comp]
    f["comp_tx"] = comp_tx[tx_comp]
    f["comp_usd"] = comp_usd[tx_comp]
    f["comp_density"] = f["comp_tx"] / np.maximum(f["comp_nodes"], 1)
    f["comp_is_small"] = (f["comp_nodes"] < 50).astype(int)

    try:
        import networkx as nx
        G = nx.DiGraph()
        agg = df.groupby([df["sender"], df["receiver"]])["usd_sent"].agg(["sum", "size"])
        G.add_weighted_edges_from(
            ((a, b, float(w)) for (a, b), w in zip(agg.index, agg["sum"].to_numpy())))
        pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=60, tol=1e-6)
        U = G.to_undirected()
        core = nx.core_number(nx.Graph(U))
        clus = nx.clustering(nx.Graph(U))

        for col, src in (("pagerank", pr), ("core", core), ("clustering", clus)):
            f[f"snd_{col}"] = df["sender"].map(src).to_numpy()
            f[f"rcv_{col}"] = df["receiver"].map(src).to_numpy()

        # Reciprocity / short cycles -- money returning to its origin.
        succ = {n: set(G.successors(n)) for n in G.nodes}
        recip, cyc3 = np.zeros(len(df)), np.zeros(len(df))
        for i, (a, b) in enumerate(zip(senders, receivers)):
            sb = succ.get(b, ())
            if a in sb:
                recip[i] = 1
            elif len(sb) < 200 and any(a in succ.get(m, ()) for m in sb):
                cyc3[i] = 1
        f["edge_reciprocal"] = recip
        f["in_3cycle"] = cyc3

        try:
            communities = nx.community.louvain_communities(
                nx.Graph(U), weight="weight", seed=SEED, resolution=1.0)
            cmap = {n: i for i, cset in enumerate(communities) for n in cset}
            comm = df["sender"].map(cmap).fillna(-1).astype(int).to_numpy()
            sizes = pd.Series(comm).map(pd.Series(comm).value_counts()).to_numpy()
            f["comm_tx"] = sizes
            f["same_community"] = (df["sender"].map(cmap).to_numpy()
                                   == df["receiver"].map(cmap).to_numpy()).astype(int)
        except Exception:
            comm = tx_comp
            f["comm_tx"] = f["comp_tx"]
            f["same_community"] = 1
    except ImportError:
        log("  networkx not installed -- skipping centrality (pip install networkx for +AP)")
        comm = tx_comp

    # CV groups: prefer components, but if one giant component swallows the
    # graph, grouping by it would collapse CV to a single fold.
    groups = tx_comp.copy()
    if comp_tx.max() > 0.5 * len(df):
        groups = np.asarray(comm)
        log("  giant component detected -> grouping CV folds by community")
    return f, groups


def f_regularity(df: pd.DataFrame) -> pd.DataFrame:
    """Machine-generated motifs leave fingerprints humans don't: evenly split
    amounts and metronomic spacing."""
    f = pd.DataFrame(index=df.index)
    g = df.sort_values(["sender", "TIMESTAMP"]).groupby("sender")
    gap = g["TIMESTAMP"].diff()
    stats = pd.DataFrame({"gap": gap, "sender": df.sort_values(["sender", "TIMESTAMP"])["sender"]})
    gs = stats.groupby("sender")["gap"]
    reg = (gs.std() / gs.mean().replace(0, np.nan)).rename("gap_cv")
    f["snd_gap_cv"] = reg.reindex(df["sender"]).to_numpy()
    f["snd_gap_mean"] = gs.mean().reindex(df["sender"]).to_numpy()

    # How many of the sender's payments share this payment's amount (±1%)?
    amt = np.round(np.log1p(np.clip(df["usd_sent"].to_numpy(), 0, None)) * 100).astype(int)
    tmp = pd.DataFrame({"sender": df["sender"].to_numpy(), "amt": amt,
                        "receiver": df["receiver"].to_numpy()})
    f["snd_amount_twins"] = tmp.groupby(["sender", "amt"])["amt"].transform("size").to_numpy()
    f["rcv_amount_twins"] = tmp.groupby(["receiver", "amt"])["amt"].transform("size").to_numpy()
    f["global_amount_twins"] = tmp.groupby("amt")["amt"].transform("size").to_numpy()
    return f


def compute_features(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    d = load_split(data_dir)
    df = build_edges(d["tx"], d["tf"], d["rates"])
    log(f"  edges built: {len(df):,} transactions, "
        f"{df['sender'].nunique():,} senders, {df['receiver'].nunique():,} receivers")

    wallets = pd.Index(pd.unique(np.concatenate([df["sender"].to_numpy(), df["receiver"].to_numpy()])))
    wmap = {a: i for i, a in enumerate(wallets)}
    s_code = df["sender"].map(wmap).to_numpy()
    r_code = df["receiver"].map(wmap).to_numpy()
    ts = df["TIMESTAMP"].to_numpy().astype(np.int64)
    usd = df["usd_sent"].to_numpy()

    # Pass 1: receiver-side fan-in degree, used as `aux` so that a second pass
    # can ask "does the wallet downstream of here sit at the mouth of a funnel?"
    idx_in_0 = WindowIndex(r_code, ts, s_code, usd)
    w6 = FOLLOW_WINDOWS["6h"]
    dest_fanin = idx_in_0.distinct(*idx_in_0.bounds(r_code, ts, w6, w6))

    idx_out = WindowIndex(s_code, ts, r_code, usd, aux=dest_fanin)
    idx_in = WindowIndex(r_code, ts, s_code, usd, aux=dest_fanin)

    blocks = [
        ("intrinsic", f_intrinsic(df)),
        ("wallet", f_wallet_profile(df)),
        ("windows", f_windows(df, idx_out, idx_in, s_code, r_code, ts)),
        ("passthrough", f_passthrough(df, idx_out, idx_in, s_code, r_code, ts)),
        ("regularity", f_regularity(df)),
    ]
    gf, groups = f_graph(df)
    blocks.append(("graph", gf))
    for name, b in blocks:
        log(f"  features[{name}]: {b.shape[1]}")

    X = pd.concat([b for _, b in blocks], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X["dest_fanin_6h"] = dest_fanin.astype("float32")
    log(f"  total features: {X.shape[1]}")
    return df, X, groups


# ----------------------------------------------------------------------------
# 5. MODEL
# ----------------------------------------------------------------------------

def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=25,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0,
            random_state=SEED, early_stopping=False)


def cv_fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Component-grouped CV. A scenario's transactions never straddle folds, so
    OOF scores are honest and safe to recycle as stage-2 inputs."""
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

    try:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y, groups))
    except Exception:
        splits = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))

    oof = np.zeros(len(y))
    models = []
    for k, (tr, va) in enumerate(splits, 1):
        m = make_model()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
        log(f"    fold {k}/{len(splits)}: {len(tr):,} train / {len(va):,} val")
    return models, oof


def predict_ens(models, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def neighbour_scores(df: pd.DataFrame, score: np.ndarray, comm: np.ndarray | None) -> pd.DataFrame:
    """STAGE 2 -- guilt by association.

    No transaction launders money alone. If the wallets around this one are
    scoring highly, this one deserves a second look even when its own shape is
    unremarkable. Self is excluded from every aggregate so a row can never
    simply re-read its own stage-1 score.
    """
    s = pd.Series(score, index=df.index)
    out = pd.DataFrame(index=df.index)

    def agg(keys: pd.Series, tag: str):
        g = s.groupby(keys)
        tot, cnt = g.transform("sum"), g.transform("count")
        out[f"nb_{tag}_mean"] = ((tot - s) / (cnt - 1).clip(lower=1)).to_numpy()
        out[f"nb_{tag}_n"] = cnt.to_numpy()
        # max of others, via the top-two trick
        mx = g.transform("max")
        tmp = pd.DataFrame({"k": keys.to_numpy(), "v": score})
        second = tmp.groupby("k")["v"].transform(lambda x: x.nlargest(2).min() if len(x) > 1 else 0.0)
        out[f"nb_{tag}_max"] = np.where(score >= mx.to_numpy(), second.to_numpy(), mx.to_numpy())

    agg(df["sender"], "snd")
    agg(df["receiver"], "rcv")
    if comm is not None:
        agg(pd.Series(comm, index=df.index), "comm")

    # Directional propagation: what does the wallet downstream of me look like,
    # and what does the wallet that funded me look like?
    by_sender = s.groupby(df["sender"]).mean()
    by_receiver = s.groupby(df["receiver"]).mean()
    out["nb_downstream"] = df["receiver"].map(by_sender).fillna(0.0).to_numpy()
    out["nb_upstream"] = df["sender"].map(by_receiver).fillna(0.0).to_numpy()
    out["nb_chain"] = out["nb_downstream"] * out["nb_upstream"]
    return out.astype("float32")


def report(y: np.ndarray, score: np.ndarray, df: pd.DataFrame, labels: pd.DataFrame | None, tag: str):
    from sklearn.metrics import average_precision_score, roc_auc_score

    ap = average_precision_score(y, score)
    auc = roc_auc_score(y, score)
    k = max(1, int(round(REVIEW_FRAC * len(y))))
    order = np.lexsort((df["TRANSACTION_HASH"].to_numpy(), -score))[:k]
    prec = y[order].mean()
    rec = y[order].sum() / max(y.sum(), 1)
    log(f"  [{tag}] AP={100*ap:5.2f}  ROC-AUC={auc:.3f}  P@1%={100*prec:5.1f}%  R@1%={100*rec:5.1f}%")

    if labels is not None and "difficulty" in labels.columns:
        d = labels.set_index("TRANSACTION_HASH")["difficulty"].reindex(df["TRANSACTION_HASH"])
        caught = pd.Series(False, index=df.index)
        caught.iloc[order] = True
        for lvl, grp in d.groupby(d.values):
            m = (d.values == lvl) & (y == 1)
            if m.sum():
                log(f"      {str(lvl):18s} recall@1% = {100*caught.to_numpy()[m].mean():5.1f}%  (n={m.sum()})")
    return ap


# ----------------------------------------------------------------------------
# 6. PIPELINE
# ----------------------------------------------------------------------------

def reason_for(row) -> str:
    bits = []
    if row.get("gs_6h_score", 0) > 6:
        bits.append("collects from many wallets then disperses similar value")
    if row.get("sg_6h_score", 0) > 6:
        bits.append("disperses to wallets that reconverge downstream")
    if row.get("fanin_1h_c_deg", 0) >= 8:
        bits.append(f"{int(row['fanin_1h_c_deg'])} distinct senders in 1h")
    if row.get("fanout_1h_c_deg", 0) >= 8:
        bits.append(f"{int(row['fanout_1h_c_deg'])} distinct receivers in 1h")
    if row.get("u_turn", 0) > 0:
        bits.append("funds forwarded within the hour of arrival")
    if row.get("in_3cycle", 0) > 0:
        bits.append("part of a short value cycle")
    return "; ".join(bits[:3]) or "elevated aggregate flow pattern"


def run(train_dir: str, test_dir: str | None, out_path: str, write_features: bool) -> None:
    log("=" * 72)
    log("TRAIN SPLIT")
    df_tr, X_tr, g_tr = compute_features(train_dir)

    labels = load_split(train_dir)["labels"]
    if labels is None:
        raise SystemExit("labels.csv not found next to the training data -- cannot fit.")
    lab = labels.set_index("TRANSACTION_HASH")["is_suspicious"].reindex(df_tr["TRANSACTION_HASH"])
    if lab.isna().mean() > 0.01:
        log(f"  WARNING: {lab.isna().mean():.1%} of transactions have no label")
    y = lab.fillna(0).astype(int).to_numpy()
    log(f"  positives: {y.sum():,} / {len(y):,} ({100*y.mean():.2f}%)")

    # Single-feature AP, purely diagnostic -- never feeds back into the model.
    from sklearn.metrics import average_precision_score
    singles = sorted(
        ((average_precision_score(y, np.nan_to_num(X_tr[c].to_numpy())), c) for c in X_tr.columns),
        reverse=True)[:12]
    log("  strongest single features (AP):")
    for ap, c in singles:
        log(f"      {100*ap:5.2f}  {c}")

    log("STAGE 1 -- gradient boosting, component-grouped CV")
    m1, oof1 = cv_fit(X_tr, y, g_tr)
    ap1 = report(y, oof1, df_tr, labels, "stage 1 OOF")

    log("STAGE 2 -- neighbourhood propagation of stage-1 scores")
    comm_tr = g_tr
    X_tr2 = pd.concat([X_tr, neighbour_scores(df_tr, oof1, comm_tr)], axis=1)
    m2, oof2 = cv_fit(X_tr2, y, g_tr)
    ap2 = report(y, oof2, df_tr, labels, "stage 2 OOF")

    if ap2 < 2 * y.mean():
        raise SystemExit(
            f"ABORT: held-out AP ({100*ap2:.2f}) is at the base rate -- the features "
            f"carry no signal. Refusing to write a submission.")
    log(f"  stage 2 lift over stage 1: {100*(ap2-ap1):+.2f} AP")

    try:
        imp = getattr(m2[0], "feature_importances_", None)
        if imp is not None:
            top = sorted(zip(imp, X_tr2.columns), reverse=True)[:20]
            log("  top features: " + ", ".join(c for _, c in top))
    except Exception:
        pass

    if test_dir is None:
        log("no --test given; stopping after evaluation")
        return

    log("=" * 72)
    log("TEST SPLIT")
    df_te, X_te, g_te = compute_features(test_dir)
    X_te = X_te.reindex(columns=X_tr.columns)        # exact column alignment
    s1 = predict_ens(m1, X_te)
    X_te2 = pd.concat([X_te, neighbour_scores(df_te, s1, g_te)], axis=1)
    X_te2 = X_te2.reindex(columns=X_tr2.columns)
    s2 = predict_ens(m2, X_te2)

    sub = pd.DataFrame({
        "TRANSACTION_HASH": df_te["TRANSACTION_HASH"].astype(str),
        "score": np.clip(s2, 0.0, 1.0).round(6),
    })
    # Reasons only for the review budget the site actually shows (top 1%);
    # generating 40k strings would bloat the CSV past no useful purpose.
    feats = X_te2.reset_index(drop=True)
    thresh = float(np.quantile(sub["score"].to_numpy(), 1.0 - REVIEW_FRAC))
    sub["reason"] = [reason_for(feats.iloc[i]) if sub["score"].iat[i] >= thresh else ""
                     for i in range(len(sub))]

    before = len(sub)
    sub = sub.drop_duplicates(subset="TRANSACTION_HASH", keep="first")
    if len(sub) != before:
        log(f"  dropped {before - len(sub)} duplicate hashes")

    n_tx = len(pd.unique(load_split(test_dir)["tx"]["TRANSACTION_HASH"]))
    if len(sub) != n_tx:
        raise SystemExit(f"ABORT: {len(sub)} rows for {n_tx} test transactions -- "
                         f"the site rejects missing or extra IDs.")

    sub.to_csv(out_path, index=False)
    log(f"wrote {out_path}: {len(sub):,} rows, "
        f"score range [{sub['score'].min():.4f}, {sub['score'].max():.4f}], "
        f"mean {sub['score'].mean():.4f}")

    if write_features:
        pd.concat([df_te[["TRANSACTION_HASH", "TIMESTAMP", "sender", "receiver", "usd_sent"]]
                   .reset_index(drop=True), X_te2], axis=1).to_csv("pattern_features.csv", index=False)
        log("wrote pattern_features.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Decode & Detect AML detector")
    p.add_argument("--train", required=True, help="folder holding train transactions + labels.csv")
    p.add_argument("--test", default=None, help="folder holding the test transactions")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--dump-features", action="store_true")
    a = p.parse_args()
    run(a.train, a.test, a.out, a.dump_features)
    log("done")


if __name__ == "__main__":
    main()
