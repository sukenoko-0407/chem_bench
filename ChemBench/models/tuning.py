from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from .builders import build_regressor
from .registry import normalize_algorithm


def _suggest_params(trial, algorithm: str) -> dict[str, Any]:
    algo = normalize_algorithm(algorithm)
    if algo == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
    if algo == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }
    if algo == "svr":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if algo == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if algo == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if algo == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if algo == "gpr":
        return {"alpha": trial.suggest_float("alpha", 1e-10, 1e-1, log=True)}
    if algo == "mlp":
        return {
            "hidden_layer_sizes": (
                trial.suggest_int("h1", 64, 256),
                trial.suggest_int("h2", 16, 128),
            ),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }
    raise ValueError(f"Unsupported algorithm for tuning: {algorithm}")


def tune_hyperparameters(
    algorithm: str,
    X: np.ndarray,
    y: np.ndarray,
    base_params: dict[str, Any],
    random_state: int,
    n_trials: int = 20,
    timeout: int | None = None,
    cv_splits: int = 5,
    use_gpu: bool = False,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required when tuning=True") from exc

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def objective(trial) -> float:
        tuned = dict(base_params)
        tuned.update(_suggest_params(trial, algorithm))
        fold_mses: list[float] = []
        for train_idx, valid_idx in kf.split(X):
            model = build_regressor(
                algorithm=algorithm,
                params=tuned,
                random_state=random_state,
                use_gpu=use_gpu,
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[valid_idx])
            fold_mses.append(mean_squared_error(y[valid_idx], pred))
        return float(np.mean(fold_mses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = dict(base_params)
    best.update(study.best_trial.params)
    if "h1" in best and "h2" in best:
        best["hidden_layer_sizes"] = (best.pop("h1"), best.pop("h2"))
    return best

