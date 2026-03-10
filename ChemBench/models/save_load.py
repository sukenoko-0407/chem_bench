from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, out_dir: Path, prefer_pickle: bool = True) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "model.pkl"
    joblib_path = out_dir / "model.joblib"

    if prefer_pickle:
        try:
            with pkl_path.open("wb") as f:
                pickle.dump(model, f)
            return "pickle"
        except Exception:
            joblib.dump(model, joblib_path)
            return "joblib"

    try:
        joblib.dump(model, joblib_path)
        return "joblib"
    except Exception:
        with pkl_path.open("wb") as f:
            pickle.dump(model, f)
        return "pickle"


def load_model(model_dir: Path) -> Any:
    pkl_path = model_dir / "model.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    joblib_path = model_dir / "model.joblib"
    if joblib_path.exists():
        return joblib.load(joblib_path)
    raise FileNotFoundError(f"No model artifact found in {model_dir}")

