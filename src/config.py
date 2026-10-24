"""Paths, seeds, windows and caps shared by every pipeline stage.

Nothing in here is learned from labels. All values are documented in
``docs/METHODOLOGY.md``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SUBMISSION = ROOT / "submission"
DOCS = ROOT / "docs"

# Candidate data folders, first existing one wins. The organiser layout keeps
# labels.csv inside train/.
TRAIN_CANDIDATES = (
    ROOT / "data" / "decode-and-detect-training-data" / "train",
    ROOT / "data" / "train",
    Path(r"C:\Users\anass\Downloads\decode-and-detect-training-data\train"),
)
TEST_CANDIDATES = (
    ROOT / "data" / "test-data",
    ROOT / "data" / "test-data" / "test-data",
)
VALIDATION_CANDIDATES = (
    ROOT / "data" / "decode-and-detect-training-data" / "validation",
    ROOT / "data" / "validation",
)

EXPECTED_TEST_ROWS = 40_101

SEED = 42

# XRPL semantics -----------------------------------------------------------
XRP_SENTINEL = "0x1111222211112222111122221111222211112222"
XRP_ASSET_ID = f"{XRP_SENTINEL}:XRP"
XRP_RATE_KEY = "XRP"  # market_rates.csv uses the bare symbol for XRP
DROPS_PER_XRP = 1_000_000
MICRO_USD = 1_000_000
RIPPLE_EPOCH_OFFSET = 946_684_800  # only needed if raw Ripple seconds appear

# Windows (seconds) ---------------------------------------------------------
WINDOWS: dict[str, int] = {
    "10m": 10 * 60,
    "1h": 3600,
    "6h": 6 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
}

# Value-aware matching -------------------------------------------------------
EPS = 1e-9
VALUE_MATCH_SCALE = 0.20          # quality = exp(-|log(out/in)| / scale)
PASSTHROUGH_TOL = 0.10            # +-10% counts as "same money moving on"
DUST_USD = 1.0                    # below this a payment is dust for motif search
MAX_SLICE = 4000                  # bound per-row slice length around hubs
MAX_CANDIDATES_PER_HOP = 12       # beam width for cycle / chain search
MAX_CYCLE_HOPS = 5
MAX_CHAIN_HOPS = 6
CYCLE_WINDOWS = {"24h": 24 * 3600, "7d": 7 * 24 * 3600}

# Model ---------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.035,
    num_leaves=31,
    min_child_samples=60,
    reg_lambda=5.0,
    colsample_bytree=0.85,
    subsample=0.9,
    subsample_freq=1,
    n_estimators=2500,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
LGBM_EARLY_STOPPING = 100
N_FOLDS = 3
REVIEW_FRAC = 0.01

GNN_PARAMS = dict(hidden=64, layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                  epochs=30, patience=7)
GNN_BLEND_WEIGHTS = (0.0, 0.15, 0.30, 0.50)

FORBIDDEN_FEATURE_PIECES = (
    "transaction_hash", "wallet", "address", "scenario", "cohort", "family",
    "difficulty", "is_suspicious", "label",
)


def first_existing(candidates) -> Path:
    for c in candidates:
        if Path(c).exists() and (Path(c) / "transactions.parquet").exists():
            return Path(c)
    raise FileNotFoundError(f"none of {list(map(str, candidates))} holds transactions.parquet")
