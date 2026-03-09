# ChemMultiBench 要件定義書 v1.0

## 1. 目的
- Pythonパッケージ `ChemMultiBench` を構築する。
- 低分子化合物のProperty予測（回帰）を、複数のClassicalアルゴリズムで一括実行できるようにする。

## 2. スコープ
- 対象タスクは回帰に限定する（分類は対象外）。
- 入力はCSVとし、`SMILES`列および`Label`列を受け取る。
- 各列名は実行時引数で指定可能にする。

## 3. 入出力仕様

### 3.1 入力
- 学習入力: CSV
  - 必須列: SMILES列、目的変数（Label）列
  - 列名は引数指定
- 予測入力: CSV
  - 必須列: SMILES列
  - 列名は引数指定

### 3.2 出力
- 学習時出力（同一の出力Dir配下に保存）
  - 学習済みモデル
  - OOF予測結果（CSV）
  - 実行時設定・評価結果サマリ（JSON）
- アルゴリズムごとにサブDirを作成して保存してよい。

## 4. 前処理仕様（SMILES）
- RDKit等を用いてSMILESのValid/Invalid判定のみ実施する。
- Invalid SMILESを1件でも検出した場合、処理を停止する。
- Validity以外の前処理チェック（標準化や重複除去など）は本パッケージの必須要件外とする。

## 5. 特徴量生成仕様
最初の処理はSMILESから記述子への変換とし、以下を選択可能にする。

1. ECFP4（1024 bit）
2. ECFP4（2048 bit）
3. その他Fingerprint（`MACCS + AtomPair`）
4. 2 と 3 の結合特徴量
5. Mordred
6. 2 と 5 の結合特徴量

### 5.1 Mordredの推奨仕様（本書での決定）
- 方針: 「基本的に限定しない」要望に合わせ、広範囲の記述子を扱う。
- 推奨デフォルト: Mordredの全2D記述子を計算対象にする。
- 3D記述子: オプション（明示指定時のみ有効化）。
  - 理由: 3Dはコンフォマー生成等の追加前処理依存が強く、安定運用の観点からデフォルト無効が妥当。

## 6. 学習アルゴリズム
以下の回帰アルゴリズムを実装対象とする。

- Ridge Regression
- Elastic Net
- SVR
- k-NN Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gaussian Process Regressor
- MLP Regressor

使用ライブラリ:
- `scikit-learn`
- `xgboost`
- `lightgbm`

## 7. 学習・評価仕様
- 検証方式: Random Splitによる5-fold CV
- OOF予測: 必須（必ず出力）
- 評価指標（CV/OOFで算出）
  - Pearson Correlation
  - Spearman Correlation
  - R2
  - MAE
  - RMSE
- 乱数seed
  - デフォルト値: `42`
  - configで上書き指定可能

## 8. ハイパーパラメータ管理
- JSON形式のデフォルトconfigファイルを提供する。
- 引数でconfig未指定時はデフォルトconfigで学習を実行する。
- MLP（`scikit-learn` `MLPRegressor`）のデフォルトは2層のコンパクト構成とする。
  - `hidden_layer_sizes: [128, 64]`
  - `activation: relu`
  - `solver: adam`
  - `alpha: 1e-4`
  - `learning_rate_init: 1e-3`
  - `batch_size: auto`
  - `max_iter: 300`
  - `early_stopping: true`
  - `n_iter_no_change: 20`
  - `random_state: 42`

### 8.1 自動チューニング（オプション）
- Optuna等を用いたハイパーパラメータ探索機能を提供する。
- 既定仕様
  - 目的関数: MSE最小化
  - 試行回数: 20
  - サンプラー: 指定なし（デフォルト）

## 9. モデル保存仕様
- 優先保存形式
  1. `pickle`（第一希望）
  2. `joblib`（pickle利用不可時の代替）
- OOF保存形式: `csv`
- 設定・結果サマリ保存形式: `json`

## 10. API/Wrapper仕様
- `fit` wrapperを提供する。
  - アルゴリズム指定あり: 指定アルゴリズムのみ学習
  - アルゴリズム指定なし: 全アルゴリズムを一括学習
- `predict` wrapperを提供する。
  - `predict(dir=...)` で指定Dir配下の全アルゴリズムモデルを自動検出し、一括予測する。

## 11. XGBoost/LightGBMの実行デバイス
- GPU対応を要件に含める。
- デフォルトはCPU実行とする。
- configでGPU実行へ切替可能とする。

## 12. 環境構築要件
- `environment.yml` を提供する。
- condaチャネルは `conda-forge` を基本とし、`defaults` チャネルの利用は禁止する。
- ただし、conda環境内での `pip install` は許可する。

## 13. 非機能要件（最低限）
- 主要処理で再現可能性を担保する（seed固定、設定保存）。
- エラーメッセージは原因が判別できる内容にする（例: Invalid SMILESの行番号・値）。
- 学習・予測の実行ログを追跡可能にする（開始/終了、使用特徴量、使用アルゴリズム、保存先）。

## 14. 今後の拡張余地（参考）
- 分類タスク対応（本v1.0の対象外）
- 外部テストセット評価レポート機能
- アンサンブル出力（平均・重み付き平均）

## 15. ディレクトリ構成・ファイル構成（Current Dir直下に構築）
- 本パッケージは `ChemMultiBench` のCurrent Dir直下に構築する。
- v1.0時点の標準構成（案）は以下とする。

```text
ChemMultiBench/
  requirements_v1.0.md
  README.md
  pyproject.toml
  environment.yml
  .gitignore
  ChemMultiBench/
    __init__.py
    cli.py
    config/
      default_config.json
      tuning_config.json
    data/
      schema.py
      io.py
      validation.py
    features/
      featurizer.py
      ecfp.py
      fingerprint.py
      mordred_desc.py
      combine.py
    models/
      registry.py
      builders.py
      train.py
      predict.py
      save_load.py
      tuning.py
    metrics/
      regression.py
    utils/
      logger.py
      random_seed.py
      paths.py
  tests/
    test_validation.py
    test_featurizer.py
    test_train_predict.py
  examples/
    sample_train.csv
    sample_predict.csv
    run_fit.py
    run_predict.py
```

### 15.1 主要ファイルの責務
- `pyproject.toml`: パッケージ定義、依存関係、ビルド設定。
- `environment.yml`: conda-forge限定の実行環境定義（必要に応じてpip依存を追記）。
- `ChemMultiBench/cli.py`: `fit` / `predict` のCLIエントリポイント。
- `ChemMultiBench/config/default_config.json`: デフォルト学習条件（各アルゴリズム、seed、CV、保存形式）。
- `ChemMultiBench/config/tuning_config.json`: Optuna利用時の探索設定（MSE最小化、trial数など）。
- `ChemMultiBench/data/validation.py`: SMILES valid/invalid検査（invalid検出時に停止）。
- `ChemMultiBench/features/`: ECFP/MACCS/AtomPair/Mordred生成と特徴量結合。
- `ChemMultiBench/models/builders.py`: 各回帰器（Ridge/ElasticNet/SVR/k-NN/XGBoost/LightGBM/GPR/MLP）のインスタンス生成。
- `ChemMultiBench/models/train.py`: 5-fold CV学習、OOF作成、評価指標算出、成果物保存。
- `ChemMultiBench/models/predict.py`: 学習済みモデルの自動検出と一括予測。
- `ChemMultiBench/models/save_load.py`: pickle優先、joblib代替の保存・読込処理。
- `ChemMultiBench/models/tuning.py`: Optunaによるハイパーパラメータ探索。
- `ChemMultiBench/metrics/regression.py`: Pearson/Spearman/R2/MAE/RMSEの計算。
- `tests/`: 前処理、特徴量生成、学習・予測フローの単体/結合テスト。
