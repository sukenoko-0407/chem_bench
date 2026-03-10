from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdMolDescriptors


def maccs_atompair(smiles_list: list[str], atom_pair_nbits: int = 2048) -> np.ndarray:
    maccs_len = 167
    total_dim = maccs_len + atom_pair_nbits
    feats = np.zeros((len(smiles_list), total_dim), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((maccs_len,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

        atompair = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol, nBits=atom_pair_nbits
        )
        ap_arr = np.zeros((atom_pair_nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(atompair, ap_arr)
        feats[i, :] = np.concatenate([maccs_arr, ap_arr], axis=0)
    return feats

