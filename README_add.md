# ChemMultiBench

ChemMultiBench は、低分子化合物の物性予測（回帰）を対象に、複数の Classical アルゴリズムを一括で学習・推論する Python パッケージです。

## 1. 主な機能
- 入力は CSV（SMILES 列・Label 列の列名は引数で指定）
- SMILES の valid/invalid を検証（invalid が1件でもあれば停止）
- 記述子を切り替えて学習可能
  - `ecfp4_1024`
  - `ecfp4_2048`（デフォルト）
  - `maccs_atompair`
  - `ecfp4_2048_plus_maccs_atompair`
  - `mordred`（デフォルトは2D）
  - `ecfp4_2048_plus_mordred`
- 学習アルゴリズム
  - `ridge`, `elastic_net`, `svr`, `knn`, `xgboost`, `lightgbm`, `gpr`, `mlp`
- 5-fold CV（Random Split）と OOF 出力を標準実行
- 評価指標を保存
  - Pearson, Spearman, R2, MAE, RMSE, MSE
- モデルは `pickle` 優先保存（失敗時 `joblib`）
- Optuna によるチューニング（オプション）

## 2. セットアップ
```bash
conda env create -f environment.yml
conda activate chemmultibench
pip install -e . --no-deps
```

`--no-deps` を付ける理由:
- 本プロジェクトは `conda-forge` で依存を入れる前提です。
- これを付けないと `pip` が `numpy` などをソースビルドしようとして、Windowsでコンパイラ不足エラーになることがあります。

## 3. CLI: fit（メイン機能）
実運用では `chemmultibench fit` の利用が中心です。
注: 以下の複数行コマンドは bash 形式です。PowerShell では行末を `` ` `` にするか、1行で実行してください。

### 3.1 最小実行（まずはこれ）
```bash
chemmultibench fit \
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

### 3.2 Mordred 記述子で学習
```bash
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_mordred \
  --smiles-col SMILES \
  --label-col Label \
  --feature-set mordred
```

### 3.3 特定アルゴリズムだけ実行
```bash
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_subset \
  --smiles-col SMILES \
  --label-col Label \
  --algorithms ridge,mlp,lightgbm
```

### 3.4 Optuna チューニングを有効化
```bash
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_tuning \
  --smiles-col SMILES \
  --label-col Label \
  --tuning
```

試行回数などを変更する場合は、`tuning_config.json` を上書きしたファイルを作って指定します。
```bash
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_tuning_custom \
  --smiles-col SMILES \
  --label-col Label \
  --tuning \
  --tuning-config-path path/to/my_tuning_config.json
```

### 3.5 カスタム設定（seed / GPU / パラメータ）
`default_config.json` をベースに上書き JSON を作り、`--config-path` で指定します。

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
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_gpu \
  --smiles-col SMILES \
  --label-col Label \
  --config-path path/to/gpu_config.json
```

例: Mordred 3D を有効化
```json
{
  "feature_options": {
    "mordred_use_3d": true
  }
}
```

### 3.6 主要オプション一覧（fit）
| 引数 | 必須 | デフォルト | 説明 | 例 |
|---|---|---|---|---|
| `--train-csv` | 必須 | なし | 学習用CSVのパス | `examples/sample_train.csv` |
| `--output-dir` | 必須 | なし | 学習結果の出力先ディレクトリ | `artifacts/run_default` |
| `--smiles-col` | 必須 | なし | SMILES列名 | `SMILES` |
| `--label-col` | 必須 | なし | 目的変数列名 | `Label` |
| `--feature-set` | 任意 | `ecfp4_2048` | 使用する記述子セット | `mordred` |
| `--algorithms` | 任意 | 未指定時は全アルゴリズム | 実行アルゴリズムをカンマ区切りで指定 | `ridge,mlp,lightgbm` |
| `--config-path` | 任意 | なし | 学習設定JSON（`default_config.json`への上書き） | `path/to/my_config.json` |
| `--tuning` | 任意 | `false` | 付与するとOptunaチューニングを有効化 | `--tuning` |
| `--tuning-config-path` | 任意 | なし | チューニング設定JSON（試行回数など） | `path/to/my_tuning_config.json` |

### 3.7 出力物
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

### 3.8 データ規模別の推奨config（追加済み）
`ChemMultiBench/config/` に、精度重視の3条件（`alpha` / `beta` / `gamma`）を用意しています。

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
- 数万件想定のため、`svr` / `knn` / `gpr` は `enabled=false` に設定
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
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_alpha_cpu \
  --smiles-col SMILES \
  --label-col Label \
  --config-path ChemMultiBench/config/condition_alpha_cpu.json
```

GPU利用例（大規模）:
```bash
chemmultibench fit \
  --train-csv examples/sample_train.csv \
  --output-dir artifacts/run_beta_gpu_large \
  --smiles-col SMILES \
  --label-col Label \
  --config-path ChemMultiBench/config/condition_beta_gpu_large.json
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

```bash
chemmultibench predict \
  --input-csv examples/sample_predict.csv \
  --model-dir artifacts/run_default \
  --smiles-col SMILES \
  --output-csv artifacts/run_default/preds.csv
```

特定アルゴリズムのみ予測したい場合:
```bash
chemmultibench predict \
  --input-csv examples/sample_predict.csv \
  --model-dir artifacts/run_default \
  --smiles-col SMILES \
  --output-csv artifacts/preds_subset.csv \
  --algorithms ridge,mlp
```

## 5. Python API（必要時）
```python
from ChemMultiBench import fit, predict

fit(
    train_csv="examples/sample_train.csv",
    output_dir="artifacts/run_api",
    smiles_col="SMILES",
    label_col="Label",
    feature_set="mordred",
    algorithms=["ridge", "mlp"],
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
