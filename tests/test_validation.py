import pytest

from ChemMultiBench.data.validation import validate_smiles


def test_validate_smiles_ok():
    validate_smiles(["CCO", "c1ccccc1"])


def test_validate_smiles_invalid():
    with pytest.raises(ValueError):
        validate_smiles(["CCO", "INVALID_SMILES"])

