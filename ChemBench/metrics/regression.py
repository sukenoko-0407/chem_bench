from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    try:
        pearson = float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        pearson = float("nan")
    try:
        spearman = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        spearman = float("nan")

    mse = mean_squared_error(y_true, y_pred)
    return {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
    }

