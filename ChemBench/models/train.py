from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..data.io import read_csv, write_csv
from ..data.validation import validate_columns, validate_smiles
from ..features.featurizer import featurize_smiles
from ..metrics.regression import regression_metrics
from ..utils.config import load_merged_config, save_json
from ..utils.logger import get_logger
from ..utils.paths import ensure_dir
from ..utils.random_seed import set_global_seed
from .builders import build_regressor
from .registry import ALGORITHMS, normalize_algorithm
from .save_load import save_model
from .tuning import tune_hyperparameters


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default_config.json"


def _default_tuning_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "tuning_config.json"


def _pca_captured_variance(model) -> float | None:
    named_steps = getattr(model, "named_steps", None)
    if not named_steps or "pca" not in named_steps:
        return None
    explained = getattr(named_steps["pca"], "explained_variance_ratio_", None)
    if explained is None:
        return None
    return float(np.sum(explained))


def _resolve_algorithms(
    config: dict[str, Any], algorithms: list[str] | None
) -> list[str]:
    if algorithms:
        return [normalize_algorithm(a) for a in algorithms]
    picked: list[str] = []
    algo_cfg = config.get("algorithms", {})
    for algo in ALGORITHMS:
        if algo_cfg.get(algo, {}).get("enabled", True):
            picked.append(algo)
    return picked


def fit_models(
    train_csv: Path,
    output_dir: Path,
    smiles_col: str,
    label_col: str,
    feature_set: str = "ecfp4_2048",
    algorithms: list[str] | None = None,
    config_path: Path | None = None,
    tuning: bool = False,
    tuning_config_path: Path | None = None,
    pca_reduction: int | None = None,
    mordred_use_3d: bool = False,
) -> dict[str, Any]:
    logger = get_logger("ChemBench.fit")
    config = load_merged_config(_default_config_path(), config_path)
    tuning_config = load_merged_config(_default_tuning_config_path(), tuning_config_path)

    seed = int(config.get("seed", 42))
    cv_splits = int(config.get("cv_splits", 5))
    mordred_use_3d = bool(mordred_use_3d)
    prefer_pickle = bool(config.get("save", {}).get("prefer_pickle", True))

    selected_algorithms = _resolve_algorithms(config, algorithms)
    if not selected_algorithms:
        raise ValueError("No algorithms selected for training.")

    set_global_seed(seed)
    ensure_dir(output_dir)
    logger.info("Loading input CSV: %s", train_csv)
    df = read_csv(train_csv)
    validate_columns(df.columns.tolist(), [smiles_col, label_col])

    smiles = df[smiles_col].astype(str).tolist()
    validate_smiles(smiles)

    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        raise ValueError(f"Label column '{label_col}' contains non-numeric values.")

    logger.info("Featurizing SMILES with feature_set=%s", feature_set)
    X = featurize_smiles(
        smiles,
        feature_set=feature_set,
        mordred_use_3d=mordred_use_3d,
        random_seed=seed,
    )
    if pca_reduction is not None:
        if pca_reduction < 1:
            raise ValueError("pca_reduction must be a positive integer.")
        # During CV, PCA is fit on train folds, so cap by the smallest train fold size.
        min_train_size = X.shape[0] - int(np.ceil(X.shape[0] / cv_splits))
        max_components = min(min_train_size, X.shape[1])
        if pca_reduction > max_components:
            raise ValueError(
                f"pca_reduction={pca_reduction} exceeds allowed max components={max_components}."
            )
        logger.info(
            "PCA reduction enabled: n_components=%d (max=%d)",
            pca_reduction,
            max_components,
        )

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    run_summary: dict[str, Any] = {
        "train_csv": str(train_csv),
        "output_dir": str(output_dir),
        "smiles_col": smiles_col,
        "label_col": label_col,
        "feature_set": feature_set,
        "feature_options": {
            "mordred_use_3d": mordred_use_3d,
        },
        "pca_reduction": pca_reduction,
        "seed": seed,
        "cv_splits": cv_splits,
        "algorithms": {},
    }

    for algorithm in selected_algorithms:
        logger.info("Training algorithm=%s", algorithm)
        algo_dir = ensure_dir(output_dir / algorithm)

        algo_cfg = config.get("algorithms", {}).get(algorithm, {})
        base_params = dict(algo_cfg.get("params", {}))
        use_gpu = bool(algo_cfg.get("use_gpu", False))
        final_params = dict(base_params)

        if tuning:
            logger.info("Tuning algorithm=%s with Optuna", algorithm)
            final_params = tune_hyperparameters(
                algorithm=algorithm,
                X=X,
                y=y,
                base_params=base_params,
                random_state=seed,
                n_trials=int(tuning_config.get("n_trials", 20)),
                timeout=tuning_config.get("timeout_seconds"),
                cv_splits=cv_splits,
                use_gpu=use_gpu,
            )

        oof_pred = np.zeros(len(df), dtype=np.float64)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            logger.info("algorithm=%s fold=%d/%d", algorithm, fold, cv_splits)
            model = build_regressor(
                algorithm=algorithm,
                params=final_params,
                random_state=seed,
                use_gpu=use_gpu,
                pca_reduction=pca_reduction,
            )
            model.fit(X[train_idx], y[train_idx])
            fold_pca_variance = _pca_captured_variance(model)
            if fold_pca_variance is not None:
                logger.info(
                    "algorithm=%s fold=%d PCA captured variance=%.2f%%",
                    algorithm,
                    fold,
                    fold_pca_variance * 100,
                )
            oof_pred[valid_idx] = model.predict(X[valid_idx])

        metrics = regression_metrics(y, oof_pred)
        logger.info(
            "algorithm=%s metrics rmse=%.6f r2=%.6f",
            algorithm,
            metrics["rmse"],
            metrics["r2"],
        )

        # Final refit on all rows using selected params.
        final_model = build_regressor(
            algorithm=algorithm,
            params=final_params,
            random_state=seed,
            use_gpu=use_gpu,
            pca_reduction=pca_reduction,
        )
        final_model.fit(X, y)
        final_pca_variance = _pca_captured_variance(final_model)
        if final_pca_variance is not None:
            logger.info(
                "algorithm=%s final PCA captured variance=%.2f%%",
                algorithm,
                final_pca_variance * 100,
            )
        save_format = save_model(final_model, algo_dir, prefer_pickle=prefer_pickle)

        oof_df = df[[smiles_col, label_col]].copy()
        oof_df["oof_pred"] = oof_pred
        write_csv(oof_df, algo_dir / "oof.csv")

        model_meta = {
            "algorithm": algorithm,
            "feature_set": feature_set,
            "feature_options": {
                "mordred_use_3d": mordred_use_3d,
            },
            "pca_reduction": pca_reduction,
            "pca_captured_variance": final_pca_variance,
            "model_format": save_format,
            "params": final_params,
            "use_gpu": use_gpu,
            "seed": seed,
            "cv_splits": cv_splits,
        }
        save_json(model_meta, algo_dir / "model_meta.json")
        save_json(metrics, algo_dir / "metrics.json")

        run_summary["algorithms"][algorithm] = {
            "metrics": metrics,
            "params": final_params,
            "use_gpu": use_gpu,
            "model_format": save_format,
            "pca_captured_variance": final_pca_variance,
        }

    save_json(run_summary, output_dir / "run_summary.json")
    logger.info("Training completed. Output dir: %s", output_dir)
    return run_summary
