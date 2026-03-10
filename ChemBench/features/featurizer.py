from __future__ import annotations

import numpy as np

from .combine import concat_features
from .ecfp import ecfp4
from .fingerprint import maccs_atompair
from .mordred_desc import mordred_features

FEATURE_SETS = {
    "ecfp4_1024",
    "ecfp4_2048",
    "maccs_atompair",
    "ecfp4_2048_plus_maccs_atompair",
    "mordred",
    "ecfp4_2048_plus_mordred",
}


def featurize_smiles(
    smiles_list: list[str],
    feature_set: str,
    atom_pair_nbits: int = 2048,
    mordred_use_3d: bool = False,
    random_seed: int = 42,
) -> np.ndarray:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unsupported feature_set: {feature_set}. Supported: {sorted(FEATURE_SETS)}"
        )

    if feature_set == "ecfp4_1024":
        return ecfp4(smiles_list, n_bits=1024)
    if feature_set == "ecfp4_2048":
        return ecfp4(smiles_list, n_bits=2048)
    if feature_set == "maccs_atompair":
        return maccs_atompair(smiles_list, atom_pair_nbits=atom_pair_nbits)
    if feature_set == "ecfp4_2048_plus_maccs_atompair":
        return concat_features(
            ecfp4(smiles_list, n_bits=2048),
            maccs_atompair(smiles_list, atom_pair_nbits=atom_pair_nbits),
        )
    if feature_set == "mordred":
        return mordred_features(
            smiles_list, use_3d=mordred_use_3d, random_seed=random_seed
        )

    return concat_features(
        ecfp4(smiles_list, n_bits=2048),
        mordred_features(
            smiles_list, use_3d=mordred_use_3d, random_seed=random_seed
        ),
    )
