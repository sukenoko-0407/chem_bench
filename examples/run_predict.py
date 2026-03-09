from ChemMultiBench import predict


if __name__ == "__main__":
    pred_df = predict(
        input_csv="examples/sample_predict.csv",
        model_dir="artifacts/sample_run",
        smiles_col="SMILES",
        output_csv="artifacts/sample_predictions.csv",
    )
    print(pred_df.head())

