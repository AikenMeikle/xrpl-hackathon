from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from payments import load_payments
BASE_DIR = Path(__file__).parent

TRAIN_DIR = BASE_DIR / "train"
#change to test-data before generating CSV
TEST_DIR = BASE_DIR / "validation"

#loads the payments
def load(directory):
    return pd.DataFrame(load_payments(directory))


train = load(TRAIN_DIR)
#we're only looking at the suspicious transactions
labels = pd.read_csv(TRAIN_DIR / "labels.csv", usecols=["TRANSACTION_HASH", "is_suspicious"])
train = train.merge(labels, on="TRANSACTION_HASH")
print(f"{len(train):,} training payments, {train.is_suspicious.sum():,} suspicious")
train.head()

#creates a few rules of suspicious transactions

HOUR, TEN_MINUTES = 3600, 600


def count_in_window(times_by_account, account, t, before, after):
    """How many of an account's payments fall between t - before and t + after."""
    times = times_by_account.get(account)
    if times is None:
        return 0
    return int(np.searchsorted(times, t + after, "right") - np.searchsorted(times, t - before, "left"))


def make_features(df):
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    t = df.TIMESTAMP.astype(int).to_numpy()
    value = df.VALUE.to_numpy()
    sends = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    gets = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    f = pd.DataFrame({"TRANSACTION_HASH": df.TRANSACTION_HASH})

    # Rule: bigger payments are more interesting (log keeps huge ones in range).
    f["log_value"] = np.log1p(value)

    # Rule: how busy are the sender and the receiver over the whole week?
    f["sender_payments"] = df.FROM_ADDRESS.map(df.FROM_ADDRESS.value_counts())
    f["receiver_payments"] = df.TO_ADDRESS.map(df.TO_ADDRESS.value_counts())

    # Rule: the receiver sends money on within an hour (a "relay" or "mule").
    f["receiver_forwards_1h"] = [
        count_in_window(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(df.TO_ADDRESS, t)
    ]

    # Rule: the sender only just received money (it is passing it along).
    f["sender_received_1h_before"] = [
        count_in_window(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(df.FROM_ADDRESS, t)
    ]

    # Rule: fan-out / fan-in, many payments from one sender or into one
    # receiver within ten minutes.
    f["sender_burst_10m"] = [
        count_in_window(sends, a, ti, TEN_MINUTES, TEN_MINUTES)
        for a, ti in zip(df.FROM_ADDRESS, t)
    ]

    f["receiver_burst_10m"] = [
        count_in_window(gets, a, ti, TEN_MINUTES, TEN_MINUTES)
        for a, ti in zip(df.TO_ADDRESS, t)
    ]

    f["sender_burst_over_5"] = (
        f["sender_burst_10m"] >= 5
    )

    f["receiver_burst_over_5"] = (
        f["receiver_burst_10m"] >= 5
)

    # Rule: the receiver passes on a similar amount (within 10%) within an hour,
    # a classic sign of money being layered through an account.
    sent_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("FROM_ADDRESS")}

    def forwards_similar(account, ti, amount):
        times = sends.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti, "right"), np.searchsorted(times, ti + HOUR, "right")
        later = sent_values[account][lo:hi]
        return bool(np.any(np.abs(later - amount) <= 0.1 * amount))

    f["receiver_forwards_similar_1h"] = [
        forwards_similar(a, ti, v) for a, ti, v in zip(df.TO_ADDRESS, t, value)
    ]

    # Rule: the sender received a similar amount within the hour before.
    got_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    def received_similar(account, ti, amount):
        times = gets.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti - HOUR, "left"), np.searchsorted(times, ti, "left")
        earlier = got_values[account][lo:hi]
        return bool(np.any(np.abs(earlier - amount) <= 0.1 * amount))

    f["sender_received_similar_1h"] = [
        received_similar(a, ti, v) for a, ti, v in zip(df.FROM_ADDRESS, t, value)
    ]

    # Rule: how quickly does the receiver send money on? (minutes, capped at 60)
    def minutes_to_next_send(account, ti):
        times = sends.get(account)
        if times is None:
            return 60.0
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    f["receiver_minutes_to_forward"] = [
        minutes_to_next_send(a, ti) for a, ti in zip(df.TO_ADDRESS, t)
    ]

    # Rule: pass-through accounts send on most of what they receive over the week.
    sent_total = df.groupby("FROM_ADDRESS").VALUE.sum()
    got_total = df.groupby("TO_ADDRESS").VALUE.sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = df.FROM_ADDRESS.map(pass_through).fillna(0)
    f["receiver_pass_through"] = df.TO_ADDRESS.map(pass_through).fillna(0)

    # ---- Add your own rules here, for example: ----
    # f["round_amount"] = (value % 100 < 1)
    # f["night_time"] = (t % 86400) < 6 * 3600

    return f.astype({c: float for c in f.columns if c != "TRANSACTION_HASH"})


train_features = make_features(train)
y = train.set_index("TRANSACTION_HASH").is_suspicious.loc[train_features.TRANSACTION_HASH].to_numpy()
X = train_features.drop(columns="TRANSACTION_HASH")

###############################################################################
print("\n=== RULE ANALYSIS ===")

analysis = X.copy()
analysis["is_suspicious"] = y

for col in X.columns:

    print(f"\n----- {col} -----")

    if analysis[col].nunique() <= 20:

        print(
            analysis.groupby(col)["is_suspicious"]
            .agg(["count", "mean"])
        )

    else:

        print(
            analysis[[col, "is_suspicious"]]
            .corr(numeric_only=True)
            .iloc[0,1]
        )


X.describe().T[["mean", "max"]]


#creates a model and trains it on 80% of the data
X_fit, X_check, y_fit, y_check = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)
model.fit(X_fit, y_fit)

check_scores = model.predict_proba(X_check)[:, 1]
print(f"Score on held-back training data: {100 * average_precision_score(y_check, check_scores):.1f}")

#retrains the model
model.fit(X, y)

test = load(TEST_DIR)
test_features = make_features(test)
predictions = pd.DataFrame({
    "TRANSACTION_HASH": test_features.TRANSACTION_HASH,
    "score": model.predict_proba(test_features.drop(columns="TRANSACTION_HASH"))[:, 1],
})
predictions.to_csv("predictions.csv", index=False)
print(f"Saved {len(predictions):,} predictions to predictions.csv")
predictions.head()
