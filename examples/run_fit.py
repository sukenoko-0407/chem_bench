from ChemBench import fit


if __name__ == "__main__":
    fit(
        train_csv="examples/sample_train.csv",
        output_dir="artifacts/sample_run",
        smiles_col="SMILES",
        label_col="Label",
        feature_set="ecfp4_1024",
    )

