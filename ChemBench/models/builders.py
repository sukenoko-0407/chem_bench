from __future__ import annotations

from typing import Any

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .registry import normalize_algorithm


def _with_scaler(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _without_scaler(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_regressor(
    algorithm: str,
    params: dict[str, Any],
    random_state: int,
    use_gpu: bool = False,
):
    algo = normalize_algorithm(algorithm)

    if algo == "ridge":
        return _with_scaler(Ridge(random_state=random_state, **params))

    if algo == "elastic_net":
        return _with_scaler(ElasticNet(random_state=random_state, **params))

    if algo == "svr":
        return _with_scaler(SVR(**params))

    if algo == "knn":
        return _with_scaler(KNeighborsRegressor(**params))

    if algo == "gpr":
        return _with_scaler(GaussianProcessRegressor(random_state=random_state, **params))

    if algo == "mlp":
        return _with_scaler(MLPRegressor(random_state=random_state, **params))

    if algo == "xgboost":
        from xgboost import XGBRegressor

        merged = dict(params)
        merged.setdefault("objective", "reg:squarederror")
        merged.setdefault("random_state", random_state)
        merged.setdefault("tree_method", "hist")
        merged["device"] = "cuda" if use_gpu else "cpu"
        return _without_scaler(XGBRegressor(**merged))

    if algo == "lightgbm":
        from lightgbm import LGBMRegressor

        merged = dict(params)
        merged.setdefault("random_state", random_state)
        merged.setdefault("objective", "regression")
        merged.setdefault("verbosity", -1)
        merged["device_type"] = "gpu" if use_gpu else "cpu"
        return _without_scaler(LGBMRegressor(**merged))

    raise ValueError(f"Unsupported algorithm: {algorithm}")

