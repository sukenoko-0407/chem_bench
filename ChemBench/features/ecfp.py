from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def ecfp4(smiles_list: list[str], n_bits: int) -> np.ndarray:
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        bitvect = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        feats[i, :] = arr
    return feats
