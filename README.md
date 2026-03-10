# ChemBench

ChemBench は、低分子化合物の物性予測（回帰）を対象に、複数の Classical アルゴリズムを一括で学習・推論する Python パッケージです。

## 1. 主な機能
- 入力は CSV（SMILES 列・Label 列の列名は引数で指定）
- SMILES の valid/invalid を検証（invalid が1件でもあれば停止）
- 記述子を切り替えて学習可能
  - `ecfp4_1024`
  - `ecfp4_2048`（デフォルト）
  - `mordred`（デフォルトは2D）
  - `ecfp4_2048_plus_mordred`
- 学習アルゴリズム
  - `ridge`, `elastic_net`, `random_forest`, `svr`, `knn`, `xgboost`, `lightgbm`, `gpr`, `mlp`
- 5-fold CV（Random Split）と OOF 出力を標準実行
- `--pca_reduction` で次元圧縮して学習可能（PCA寄与率をログ出力）
- 評価指標を保存
  - Pearson, Spearman, R2, MAE, RMSE, MSE
- モデルは `pickle` 優先保存（失敗時 `joblib`）
- Optuna によるチューニング（オプション）

## 2. セットアップ
```bash
conda env create -f environment.yml
conda activate chembench
pip install -e . 
```

## 2.1 引数一覧（先に確認）
### fit の主要引数
| 引数 | 必須 | デフォルト | 説明 | 例 |
|---|---|---|---|---|
| `--train-csv` | 必須 | なし | 学習用CSVのパス | `examples/sample_train.csv` |
| `--output-dir` | 必須 | なし | 学習結果の出力先ディレクトリ | `artifacts/run_default` |
| `--smiles-col` | 必須 | なし | SMILES列名 | `SMILES` |
| `--label-col` | 必須 | なし | 目的変数列名 | `Label` |
| `--feature-set` | 任意 | `ecfp4_2048` | 使用する記述子セット | `mordred` |
| `--mordred-3d` / `--mordred_use_3d` | 任意 | `false` | 指定時はMordred 3Dを使用（未指定は2D） | `--mordred_use_3d` |
| `--pca-reduction` / `--pca_reduction` | 任意 | なし | 記述子を指定次元へPCA圧縮してから学習 | `--pca_reduction 128` |
| `--algorithms` | 任意 | 未指定時は全アルゴリズム | 実行アルゴリズムをカンマ区切りで指定 | `ridge,random_forest,lightgbm` |
| `--config-path` | 任意 | なし | 学習設定JSON（`default_config.json`への上書き） | `path/to/my_config.json` |
| `--tuning` | 任意 | `false` | 付与するとOptunaチューニングを有効化 | `--tuning` |
| `--tuning-config-path` | 任意 | なし | チューニング設定JSON（試行回数など） | `path/to/my_tuning_config.json` |

### 記述子選択一覧（`--feature-set`）
| 値 | 概要 | 特徴量次元（目安） | 関連引数 |
|---|---|---|---|
| `ecfp4_1024` | ECFP4（Morgan, radius=2） | `1024` | なし |
| `ecfp4_2048` | ECFP4（Morgan, radius=2） | `2048` | なし（デフォルト） |
| `mordred` | Mordred 記述子 | 利用可能な記述子数に依存 | `--mordred-3d` / `--mordred_use_3d` |
| `ecfp4_2048_plus_mordred` | ECFP4(2048) + Mordred | `2048 + Mordred次元` | `--mordred-3d` / `--mordred_use_3d` |

### predict の主要引数
| 引数 | 必須 | デフォルト | 説明 | 例 |
|---|---|---|---|---|
| `--input-csv` | 必須 | なし | 予測対象CSVのパス | `examples/sample_predict.csv` |
| `--model-dir` | 必須 | なし | 学習済み成果物ディレクトリ | `artifacts/run_default` |
| `--smiles-col` | 必須 | なし | SMILES列名 | `SMILES` |
| `--output-csv` | 任意 | なし | 予測結果の保存先CSV。未指定時は保存せずDataFrameを返す | `artifacts/run_default/preds.csv` |
| `--algorithms` | 任意 | 未指定時は検出した全モデル | 予測対象アルゴリズムをカンマ区切りで指定 | `ridge,mlp` |

## 3. CLI: fit（メイン機能）
実運用では `chembench fit` の利用が中心です。

基本構文:
```bash
chembench fit --train-csv <PATH> --output-dir <DIR> --smiles-col <COL> --label-col <COL> [--feature-set <NAME>] [--mordred_use_3d] [--pca_reduction <N>] [--algorithms <CSV>] [--config-path <PATH>] [--tuning] [--tuning-config-path <PATH>]
```

### 3.1 最小実行
```bash
chembench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_default \
  --smiles-col SMILES \
  --label-col Label
```

この実行で、以下が行われます。
- 記述子: `ecfp4_2048`（デフォルト）
- アルゴリズム: 全アルゴリズムを実行
- CV: 5-fold
- 出力: `artifacts/run_default/<algorithm>/...`

#### Mordred 記述子を利用する場合
* `--feature-set mordred`を追加してください

#### 特定アルゴリズムだけで実行する場合
* `--algorithms ridge,random_forest,lightgbm`のように実行したいアルゴリズムを指定してください

#### Optuna チューニングを有効化する場合
* `--tuning`を追加してください
* 試行回数などを変更する場合は、`tuning_config.json` を上書きしたファイルを作って指定してください
  - `--tuning-config-path path/to/my_tuning_config.json`

#### カスタム設定で学習する場合
* `default_config.json` をベースに上書き JSON を作り、`--config-path` で指定してください

例: GPUを使う（XGBoost, LightGBM）
```json
{
  "algorithms": {
    "xgboost": { "use_gpu": true },
    "lightgbm": { "use_gpu": true }
  }
}
```

```bash
chembench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_gpu \
  --smiles-col SMILES \
  --label-col Label \
  --config-path path/to/gpu_config.json
```

#### Mordred 3D を有効化する場合
* `--mordred_use_3d`を追加してください

#### PCAで記述子を次元圧縮する場合
* `--pca_reduction 128`のように次元を指定してください



### 3.2 出力
`--output-dir` 配下にアルゴリズムごとのサブディレクトリが作られます。

```text
artifacts/run_xxx/
  run_summary.json
  ridge/
    model.pkl or model.joblib
    oof.csv
    metrics.json
    model_meta.json
  mlp/
    ...
```

### 3.3 データ規模別の推奨config（追加済み）
`ChemBench/config/` に、精度重視の3条件（`alpha` / `beta` / `gamma`）を用意しています。

- `alpha`: 精度重視の標準設定（高性能だが過度に攻めすぎない）
- `beta`: さらに高容量寄り（ツリー数・MLP容量を増やした重め設定）
- `gamma`: 正則化を強めた安定重視設定（汎化優先）

ファイル構成:
- CPU版
  - `condition_alpha_cpu.json`
  - `condition_beta_cpu.json`
  - `condition_gamma_cpu.json`
  - `condition_alpha_cpu_large.json`
  - `condition_beta_cpu_large.json`
  - `condition_gamma_cpu_large.json`
- GPU版（`xgboost` / `lightgbm` のみ `use_gpu=true`）
  - `condition_alpha_gpu.json`
  - `condition_beta_gpu.json`
  - `condition_gamma_gpu.json`
  - `condition_alpha_gpu_large.json`
  - `condition_beta_gpu_large.json`
  - `condition_gamma_gpu_large.json`

alpha / beta / gamma の条件一覧（通常版）:

| 条件 | ねらい | XGBoost（主） | LightGBM（主） | MLP（主） | SVR/KNN/GPR |
|---|---|---|---|---|---|
| `alpha` | 精度重視の標準 | `n_estimators=1600`, `max_depth=10`, `lr=0.03` | `n_estimators=1800`, `num_leaves=255`, `lr=0.025` | `[512,256]`, `max_iter=600`, `batch_size=256` | 有効（`C=30`, `k=11`, `gpr alpha=1e-8`） |
| `beta` | 最も攻めた高容量 | `n_estimators=2200`, `max_depth=12`, `lr=0.02` | `n_estimators=2400`, `num_leaves=511`, `lr=0.02` | `[768,384]`, `max_iter=800`, `batch_size=256` | 有効（`C=80`, `k=7`, `gpr alpha=1e-10`） |
| `gamma` | 正則化強めの安定重視 | `n_estimators=1400`, `max_depth=8`, `lr=0.02`, `reg_lambda=2` | `n_estimators=1600`, `num_leaves=255`, `lr=0.025`, `reg_lambda=2` | `[512,256]`, `max_iter=500`, `batch_size=128` | 有効（`C=20`, `k=21`, `gpr alpha=1e-6`） |

`*_large` 版の方針:
- 数万件想定のため、`svr` / `knn` / `gpr` / `random_forest` は `enabled=false` に設定
- `xgboost` / `lightgbm` / `mlp` / `ridge` / `elastic_net` を中心に精度を狙う

alpha / beta / gamma の条件一覧（`*_large` 版）:

| 条件 | 有効アルゴリズム | XGBoost（主） | LightGBM（主） | MLP（主） |
|---|---|---|---|---|
| `alpha_large` | `ridge`, `elastic_net`, `xgboost`, `lightgbm`, `mlp` | `n_estimators=2600`, `max_depth=10`, `lr=0.02` | `n_estimators=3000`, `num_leaves=255`, `lr=0.015` | `[512,256]`, `max_iter=500`, `batch_size=512` |
| `beta_large` | `ridge`, `elastic_net`, `xgboost`, `lightgbm`, `mlp` | `n_estimators=3500`, `max_depth=12`, `lr=0.015` | `n_estimators=4000`, `num_leaves=511`, `lr=0.01` | `[768,384]`, `max_iter=700`, `batch_size=512` |
| `gamma_large` | `ridge`, `elastic_net`, `xgboost`, `lightgbm`, `mlp` | `n_estimators=2200`, `max_depth=8`, `lr=0.02`, `reg_lambda=2` | `n_estimators=2600`, `num_leaves=255`, `lr=0.015`, `reg_lambda=2` | `[512,256]`, `max_iter=400`, `batch_size=1024` |

GPU版（`condition_*_gpu*.json`）は、上表と同じハイパーパラメータで、`xgboost` / `lightgbm` の `use_gpu` だけを `true` にしたものです。

CPU利用例:
```bash
chembench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_alpha_cpu \
  --smiles-col SMILES \
  --label-col Label \
  --config-path ChemBench/config/condition_alpha_cpu.json
```

GPU利用例（大規模）:
```bash
chembench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_beta_gpu_large \
  --smiles-col SMILES \
  --label-col Label \
  --config-path ChemBench/config/condition_beta_gpu_large.json
```

### 3.9 アルゴリズム別の学習データ上限目安（CPU 64コア前提）
前提:
- 単一目的変数の回帰
- 5-fold CV（本パッケージの標準）
- 特徴量は ECFP2048 〜 ECFP+Mordred 程度
- 実際の上限は RAM 容量と特徴量次元で大きく変動

| アルゴリズム | 上限目安（件数） | コメント |
|---|---:|---|
| `ridge` | 200k〜500k+ | 線形モデルで比較的スケールしやすい。 |
| `elastic_net` | 100k〜300k | 反復最適化のため `ridge` より重い。 |
| `random_forest` | 50k〜300k | 非線形を扱いやすいが、木の本数・深さに応じてメモリと時間が増える。 |
| `svr` (RBF) | 5k〜10k | カーネル法で計算量が急増。64コアでも大幅短縮しにくい。 |
| `knn` | 50k〜200k | 学習は軽いが、推論コストとメモリ負荷が大きくなる。 |
| `xgboost` | 100k〜1M+ | 大規模向け。GPU対応あり。 |
| `lightgbm` | 200k〜2M+ | 大規模向け。GPU対応あり。 |
| `gpr` | 2k〜5k | カーネル行列でメモリ/計算が厳しい。1万件は通常非現実的。 |
| `mlp` | 100k〜500k | 設定次第で伸ばせるが、収束時間とメモリに依存。 |

SVR/GPRについて:
- 本実装では `svr` と `gpr` は scikit-learn のCPU実装で、GPU経路はありません。
- そのため「GPUを有効化」しても `svr` / `gpr` の学習速度には基本的に効きません。
- 数万件では、`svr` / `knn` / `gpr` は通常は無効化を推奨します（`condition_*_large.json` と同じ方針）。

## 4. CLI: predict（簡潔版）
学習済みディレクトリ内のモデルを自動検出し、一括予測します。

基本構文:
```bash
chembench predict --input-csv <PATH> --model-dir <DIR> --smiles-col <COL> [--output-csv <PATH>] [--algorithms <CSV>]
```

```bash
chembench predict \
  --input-csv examples/sample_predict.csv \
  --model-dir artifacts/run_default \
  --smiles-col SMILES \
  --output-csv artifacts/run_default/preds.csv
```

特定アルゴリズムのみ予測したい場合:
```bash
chembench predict \
  --input-csv examples/sample_predict.csv \
  --model-dir artifacts/run_default \
  --smiles-col SMILES \
  --output-csv artifacts/preds_subset.csv \
  --algorithms ridge,mlp
```

### 4.1 主要オプション一覧（predict）
引数一覧は [2.1 引数一覧（先に確認）](#21-引数一覧先に確認) を参照してください。

## 5. Python API（必要時）
```python
from ChemBench import fit, predict

fit(
    train_csv="examples/sample_train.csv",
    output_dir="artifacts/run_api",
    smiles_col="SMILES",
    label_col="Label",
    feature_set="mordred",
    algorithms=["ridge", "random_forest", "mlp"],
)

pred_df = predict(
    input_csv="examples/sample_predict.csv",
    model_dir="artifacts/run_api",
    smiles_col="SMILES",
    output_csv="artifacts/preds_api.csv",
)
```

## 6. よくあるエラー
- `Invalid SMILES detected ...`
  - 入力CSVに無効な SMILES が含まれています。対象行を修正して再実行してください。
- `Missing required columns ...`
  - `--smiles-col` / `--label-col` の指定と CSV の列名を確認してください。
- GPU設定時に学習失敗
  - GPU対応版の実行環境（xgboost / lightgbm / ドライバ）を確認してください。
- `ImportError: cannot import name 'product' from 'numpy'`（Mordred利用時）
  - `numpy 2.x` と `mordred` の相性問題です。`numpy<2` にしてください。
  - 例:
    - `conda install -c conda-forge "numpy<2"`
    - その後 `pip install -e . --no-deps`
