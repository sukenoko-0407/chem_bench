from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.io import read_csv, write_csv
from ..data.validation import validate_columns, validate_smiles
from ..features.featurizer import featurize_smiles
from ..utils.config import load_json
from ..utils.logger import get_logger
from .registry import normalize_algorithm
from .save_load import load_model


def _discover_algorithms(model_dir: Path) -> list[str]:
    found = []
    for child in model_dir.iterdir():
        if child.is_dir() and (child / "model_meta.json").exists():
            found.append(child.name)
    return sorted(found)


def predict_from_dir(
    input_csv: Path,
    model_dir: Path,
    smiles_col: str,
    output_csv: Path | None = None,
    algorithms: list[str] | None = None,
):
    logger = get_logger("ChemBench.predict")
    logger.info("Loading input CSV: %s", input_csv)
    df = read_csv(input_csv)
    validate_columns(df.columns.tolist(), [smiles_col])
    smiles = df[smiles_col].astype(str).tolist()
    validate_smiles(smiles)

    if algorithms:
        selected = [normalize_algorithm(a) for a in algorithms]
    else:
        selected = _discover_algorithms(model_dir)
    if not selected:
        raise ValueError(f"No model directories found under: {model_dir}")

    feat_cache: dict[tuple[Any, ...], Any] = {}
    out = df.copy()

    for algorithm in selected:
        algo_dir = model_dir / algorithm
        if not algo_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {algo_dir}")
        meta = load_json(algo_dir / "model_meta.json")

        feature_set = meta["feature_set"]
        feature_options = meta.get("feature_options", {})
        cache_key = (
            feature_set,
            bool(feature_options.get("mordred_use_3d", False)),
        )

        if cache_key not in feat_cache:
            logger.info("Featurizing for feature_set=%s", feature_set)
            feat_cache[cache_key] = featurize_smiles(
                smiles_list=smiles,
                feature_set=feature_set,
                mordred_use_3d=cache_key[1],
                random_seed=int(meta.get("seed", 42)),
            )
        X = feat_cache[cache_key]

        model = load_model(algo_dir)
        pred = model.predict(X)
        out[f"pred_{algorithm}"] = pred

    if output_csv:
        write_csv(out, output_csv)
        logger.info("Predictions saved to: %s", output_csv)
    return out
