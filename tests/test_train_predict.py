import pandas as pd

from ChemMultiBench import fit, predict


def test_train_and_predict(tmp_path):
    train_csv = tmp_path / "train.csv"
    pred_csv = tmp_path / "pred.csv"
    out_dir = tmp_path / "artifacts"

    train_df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "CCS", "CCP", "CCCO", "CCCN"],
            "Label": [0.1, 0.2, 0.12, 0.3, 0.33, 0.22, 0.18, 0.24, 0.15, 0.19],
        }
    )
    test_df = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC"]})
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(pred_csv, index=False)

    fit(
        train_csv=str(train_csv),
        output_dir=str(out_dir),
        smiles_col="SMILES",
        label_col="Label",
        feature_set="ecfp4_1024",
        algorithms=["ridge"],
    )

    pred = predict(
        input_csv=str(pred_csv),
        model_dir=str(out_dir),
        smiles_col="SMILES",
        algorithms=["ridge"],
    )
    assert "pred_ridge" in pred.columns
    assert len(pred) == 3

