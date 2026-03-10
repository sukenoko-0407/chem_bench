import pytest

from ChemBench.features.featurizer import featurize_smiles


def test_ecfp_shape():
    X = featurize_smiles(["CCO", "CCN"], feature_set="ecfp4_1024")
    assert X.shape == (2, 1024)


def test_combined_shape():
    X = featurize_smiles(
        ["CCO", "CCN"],
        feature_set="ecfp4_2048_plus_mordred",
    )
    assert X.shape[0] == 2
    assert X.shape[1] > 2048


def test_removed_feature_set_raises():
    with pytest.raises(ValueError):
        featurize_smiles(["CCO"], feature_set="maccs_atompair")
