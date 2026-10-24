"""Regularised LightGBM binary classifier with AP early stopping."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
FAST = dict(n_estimators=500, learning_rate=0.08)  # for ablations / diagnostics only

from src.config import FORBIDDEN_FEATURE_PIECES, LGBM_EARLY_STOPPING, LGBM_PARAMS


def assert_no_forbidden(columns) -> None:
    bad = [c for c in columns if any(p in c.lower() for p in FORBIDDEN_FEATURE_PIECES)]
    if bad:
        raise AssertionError(f"identifier/label-like feature columns: {bad[:10]}")


def make_classifier(scale_pos_weight: float = 1.0, **overrides) -> LGBMClassifier:
    params = {**LGBM_PARAMS, **overrides}
    return LGBMClassifier(scale_pos_weight=scale_pos_weight, **params)


def fit_early_stopping(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray,
                       scale_pos_weight: float = 1.0, **overrides) -> tuple[LGBMClassifier, int, np.ndarray]:
    assert_no_forbidden(x_tr.columns)
    model = make_classifier(scale_pos_weight, **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], eval_metric="average_precision",
                  callbacks=[early_stopping(LGBM_EARLY_STOPPING, first_metric_only=True, verbose=False), log_evaluation(0)])
    best = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
    return model, best, model.predict_proba(x_va)[:, 1]


def oof_lgbm(x: pd.DataFrame, y: np.ndarray, folds, scale_pos_weight: float = 1.0, allow_partial: bool = False, **overrides) -> dict:
    oof = np.full(len(y), np.nan)
    fold_ap, iters, models = [], [], []
    for tr, va in folds:
        m, best, s = fit_early_stopping(x.iloc[tr], y[tr], x.iloc[va], y[va], scale_pos_weight, **overrides)
        oof[va] = s
        fold_ap.append(float(average_precision_score(y[va], s)))
        iters.append(best)
        models.append(m)
    if np.isnan(oof).any() and not allow_partial:
        raise AssertionError("OOF has unscored rows")
    return {"oof": oof, "fold_ap": fold_ap, "mean_ap": float(np.mean(fold_ap)), "std_ap": float(np.std(fold_ap)),
            "best_iterations": iters, "models": models}


def fit_final(x: pd.DataFrame, y: np.ndarray, n_estimators: int, scale_pos_weight: float = 1.0) -> LGBMClassifier:
    assert_no_forbidden(x.columns)
    model = make_classifier(scale_pos_weight, n_estimators=int(n_estimators))
    model.fit(x, y)
    return model


def gain_importance(model: LGBMClassifier, columns) -> pd.Series:
    return pd.Series(model.booster_.feature_importance(importance_type="gain"), index=list(columns)).sort_values(ascending=False)


def shap_contributions(model: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values from LightGBM (last column is the bias)."""
    return model.booster_.predict(x, pred_contrib=True)
