from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_merged_config(default_path: Path, custom_path: Path | None) -> dict[str, Any]:
    base = load_json(default_path)
    if custom_path is None:
        return base
    custom = load_json(custom_path)
    return _deep_update(base, custom)

