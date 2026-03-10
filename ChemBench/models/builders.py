from __future__ import annotations

from typing import Any

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .registry import normalize_algorithm


def _with_scaler(model, pca_reduction: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        # Keep all-NaN columns and impute them to 0 to avoid fold-wise feature drops.
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    if pca_reduction is not None:
        steps.append(("pca", PCA(n_components=pca_reduction, random_state=0)))
    steps.append(("model", model))
    return Pipeline(steps)


def _without_scaler(model, pca_reduction: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if pca_reduction is not None:
        steps.append(("pca", PCA(n_components=pca_reduction, random_state=0)))
    steps.append(("model", model))
    return Pipeline(steps)


def build_regressor(
    algorithm: str,
    params: dict[str, Any],
    random_state: int,
    use_gpu: bool = False,
    pca_reduction: int | None = None,
):
    algo = normalize_algorithm(algorithm)

    if algo == "ridge":
        return _with_scaler(
            Ridge(random_state=random_state, **params), pca_reduction=pca_reduction
        )

    if algo == "elastic_net":
        return _with_scaler(
            ElasticNet(random_state=random_state, **params), pca_reduction=pca_reduction
        )

    if algo == "svr":
        return _with_scaler(SVR(**params), pca_reduction=pca_reduction)

    if algo == "knn":
        return _with_scaler(KNeighborsRegressor(**params), pca_reduction=pca_reduction)

    if algo == "gpr":
        return _with_scaler(
            GaussianProcessRegressor(random_state=random_state, **params),
            pca_reduction=pca_reduction,
        )

    if algo == "mlp":
        return _with_scaler(
            MLPRegressor(random_state=random_state, **params), pca_reduction=pca_reduction
        )

    if algo == "xgboost":
        from xgboost import XGBRegressor

        merged = dict(params)
        merged.setdefault("objective", "reg:squarederror")
        merged.setdefault("random_state", random_state)
        merged.setdefault("tree_method", "hist")
        merged["device"] = "cuda" if use_gpu else "cpu"
        return _without_scaler(XGBRegressor(**merged), pca_reduction=pca_reduction)

    if algo == "lightgbm":
        from lightgbm import LGBMRegressor

        merged = dict(params)
        merged.setdefault("random_state", random_state)
        merged.setdefault("objective", "regression")
        merged.setdefault("verbosity", -1)
        merged["device_type"] = "gpu" if use_gpu else "cpu"
        return _without_scaler(LGBMRegressor(**merged), pca_reduction=pca_reduction)

    raise ValueError(f"Unsupported algorithm: {algorithm}")
