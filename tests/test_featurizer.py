from ChemMultiBench.features.featurizer import featurize_smiles


def test_ecfp_shape():
    X = featurize_smiles(["CCO", "CCN"], feature_set="ecfp4_1024")
    assert X.shape == (2, 1024)


def test_combined_shape():
    X = featurize_smiles(
        ["CCO", "CCN"],
        feature_set="ecfp4_2048_plus_maccs_atompair",
        atom_pair_nbits=2048,
    )
    assert X.shape == (2, 2048 + 167 + 2048)

