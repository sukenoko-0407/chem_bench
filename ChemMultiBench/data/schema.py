from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetColumns:
    smiles_col: str
    label_col: str | None = None

