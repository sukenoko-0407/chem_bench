from __future__ import annotations

from typing import Iterable

from rdkit import Chem


def validate_columns(columns: list[str], required: list[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_smiles(smiles_values: Iterable[str]) -> None:
    invalid: list[tuple[int, str]] = []
    for idx, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid.append((idx, str(smiles)))
    if invalid:
        preview = ", ".join([f"(row={i}, smiles='{s}')" for i, s in invalid[:10]])
        suffix = " ..." if len(invalid) > 10 else ""
        raise ValueError(
            f"Invalid SMILES detected ({len(invalid)} rows): {preview}{suffix}"
        )

