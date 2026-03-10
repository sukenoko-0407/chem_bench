from __future__ import annotations

ALGORITHMS = [
    "ridge",
    "elastic_net",
    "svr",
    "knn",
    "xgboost",
    "lightgbm",
    "gpr",
    "mlp",
]

ALIASES = {
    "ridge": "ridge",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "svr": "svr",
    "knn": "knn",
    "k-nn": "knn",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "gpr": "gpr",
    "gaussian_process_regression": "gpr",
    "mlp": "mlp",
}


def normalize_algorithm(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unsupported algorithm: {name}")
    return ALIASES[key]

