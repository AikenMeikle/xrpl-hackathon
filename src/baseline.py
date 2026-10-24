"""Organiser starter baseline (logistic regression on simple rules).

Ported from ``starter_notebook.ipynb`` so it runs as ``python run.py baseline``.
Uses the organiser ``payments.py`` projection unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ROOT

sys.path.insert(0, str(ROOT))
from payments import load_payments  # noqa: E402  organiser helper

HOUR, TEN_MINUTES = 3600, 600


def _count_in_window(times_by_account, account, t, before, after):
    times = times_by_account.get(account)
    if times is None:
        return 0
    return int(np.searchsorted(times, t + after, "right") - np.searchsorted(times, t - before, "left"))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    t = df.TIMESTAMP.astype(int).to_numpy()
    value = df.VALUE.to_numpy()
    sends = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    gets = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("TO_ADDRESS")}
    f = pd.DataFrame({"TRANSACTION_HASH": df.TRANSACTION_HASH})
    f["log_value"] = np.log1p(value)
    f["sender_payments"] = df.FROM_ADDRESS.map(df.FROM_ADDRESS.value_counts())
    f["receiver_payments"] = df.TO_ADDRESS.map(df.TO_ADDRESS.value_counts())
    f["receiver_forwards_1h"] = [_count_in_window(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(df.TO_ADDRESS, t)]
    f["sender_received_1h_before"] = [_count_in_window(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(df.FROM_ADDRESS, t)]
    f["sender_burst_10m"] = [_count_in_window(sends, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.FROM_ADDRESS, t)]
    f["receiver_burst_10m"] = [_count_in_window(gets, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.TO_ADDRESS, t)]
    sent_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    got_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    def forwards_similar(account, ti, amount):
        times = sends.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti, "right"), np.searchsorted(times, ti + HOUR, "right")
        return bool(np.any(np.abs(sent_values[account][lo:hi] - amount) <= 0.1 * amount))

    def received_similar(account, ti, amount):
        times = gets.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti - HOUR, "left"), np.searchsorted(times, ti, "left")
        return bool(np.any(np.abs(got_values[account][lo:hi] - amount) <= 0.1 * amount))

    def minutes_to_next_send(account, ti):
        times = sends.get(account)
        if times is None:
            return 60.0
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    f["receiver_forwards_similar_1h"] = [forwards_similar(a, ti, v) for a, ti, v in zip(df.TO_ADDRESS, t, value)]
    f["sender_received_similar_1h"] = [received_similar(a, ti, v) for a, ti, v in zip(df.FROM_ADDRESS, t, value)]
    f["receiver_minutes_to_forward"] = [minutes_to_next_send(a, ti) for a, ti in zip(df.TO_ADDRESS, t)]
    sent_total = df.groupby("FROM_ADDRESS").VALUE.sum()
    got_total = df.groupby("TO_ADDRESS").VALUE.sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = df.FROM_ADDRESS.map(pass_through).fillna(0)
    f["receiver_pass_through"] = df.TO_ADDRESS.map(pass_through).fillna(0)
    return f.astype({c: float for c in f.columns if c != "TRANSACTION_HASH"})


def run_baseline(train_dir: Path, test_dir: Path, output: Path) -> dict:
    train = pd.DataFrame(load_payments(train_dir))
    labels = pd.read_csv(Path(train_dir) / "labels.csv", usecols=["TRANSACTION_HASH", "is_suspicious"])
    train = train.merge(labels, on="TRANSACTION_HASH")
    feats = make_features(train)
    y = train.set_index("TRANSACTION_HASH").is_suspicious.loc[feats.TRANSACTION_HASH].to_numpy()
    X = feats.drop(columns="TRANSACTION_HASH")
    X_fit, X_check, y_fit, y_check = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X_fit, y_fit)
    ap = float(average_precision_score(y_check, model.predict_proba(X_check)[:, 1]))
    model.fit(X, y)
    test = pd.DataFrame(load_payments(test_dir))
    tf = make_features(test)
    pred = pd.DataFrame({
        "TRANSACTION_HASH": tf.TRANSACTION_HASH,
        "score": model.predict_proba(tf.drop(columns="TRANSACTION_HASH"))[:, 1],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False, float_format="%.12g")
    return {"random_holdout_ap": ap, "rows": int(len(pred)), "output": str(output)}
