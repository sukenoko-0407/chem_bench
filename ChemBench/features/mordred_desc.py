from __future__ import annotations

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem


def _build_3d_mol(smiles: str, random_seed: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError(f"3D conformer generation failed for SMILES: {smiles}")

    if AllChem.UFFHasAllMoleculeParams(mol):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def mordred_features(
    smiles_list: list[str], use_3d: bool = False, random_seed: int = 42
) -> np.ndarray:
    if use_3d:
        mols = [_build_3d_mol(s, random_seed=random_seed) for s in smiles_list]
    else:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    calculator = Calculator(descriptors, ignore_3D=not use_3d)
    desc_df = calculator.pandas(mols)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = {}
    for col in desc_df.columns:
        series = desc_df[col]
        if series.dtype == bool:
            numeric_cols[str(col)] = series.astype(np.float32)
        else:
            numeric_cols[str(col)] = pd.to_numeric(series, errors="coerce")

    out = pd.DataFrame(numeric_cols)
    return out.to_numpy(dtype=np.float32)
