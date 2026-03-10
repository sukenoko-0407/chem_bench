from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.predict import predict_from_dir
from .models.train import fit_models


def fit(
    train_csv: str,
    output_dir: str,
    smiles_col: str,
    label_col: str,
    feature_set: str = "ecfp4_2048",
    algorithms: list[str] | None = None,
    config_path: str | None = None,
    tuning: bool = False,
    tuning_config_path: str | None = None,
    pca_reduction: int | None = None,
    mordred_use_3d: bool = False,
) -> dict[str, Any]:
    """Train one or multiple algorithms and save artifacts."""
    return fit_models(
        train_csv=Path(train_csv),
        output_dir=Path(output_dir),
        smiles_col=smiles_col,
        label_col=label_col,
        feature_set=feature_set,
        algorithms=algorithms,
        config_path=Path(config_path) if config_path else None,
        tuning=tuning,
        tuning_config_path=Path(tuning_config_path) if tuning_config_path else None,
        pca_reduction=pca_reduction,
        mordred_use_3d=mordred_use_3d,
    )


def predict(
    input_csv: str,
    model_dir: str,
    smiles_col: str,
    output_csv: str | None = None,
    algorithms: list[str] | None = None,
):
    """Run prediction using all (or specified) models found in model_dir."""
    return predict_from_dir(
        input_csv=Path(input_csv),
        model_dir=Path(model_dir),
        smiles_col=smiles_col,
        output_csv=Path(output_csv) if output_csv else None,
        algorithms=algorithms,
    )
