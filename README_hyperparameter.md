# ハイパーパラメータ最適化ガイド

このドキュメントでは、ChemGLaMのハイパーパラメータ最適化の実行方法を説明します。

## 概要

`optimize.py`を使用して、Optunaによるハイパーパラメータ探索を実行できます。

### 探索対象パラメータ

- `learning_rate`: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4] = 6通り
- `dropout`: [0.1, 0.2, 0.3] = 3通り
- `mlp_num_layers`: [1, 2, 3, 4] = 4通り (MLPの層数)
- `batch_size`: [8, 16, 32, 64] = 4通り
  - batch_size > 16の場合、自動的にgradient accumulationを使用

**探索空間の総数:** 6 × 3 × 4 × 4 = **288通り**（Grid samplerの場合）

## 基本的な実行方法

### 1. 単純な実行（50試行）

```bash
python optimize.py -c config/benchmark/davis_cv0.json -n 50 \
  --study_name davis_cv0_hpo \
  --use_wandb --wandb_project "ChemGLaM-Davis-HPO"
```

### 2. データベースに保存して再開可能にする（推奨）

```bash
python optimize.py -c config/benchmark/davis_cv0.json -n 50 \
  --study_name davis_cv0_hpo \
  --storage sqlite:///davis_cv0_hpo.db \
  --use_wandb --wandb_project "ChemGLaM-Davis-HPO"
```

**メリット:**
- 全試行履歴が`davis_cv0_hpo.db`に保存される
- 同じコマンドを再実行すると、続きから探索を再開できる
- 複数のプロセスで並列実行も可能

### 3. 追加試行の実行

既存のstudyに試行を追加する場合は、同じコマンドを再実行します：

```bash
# さらに50試行追加（合計100試行）
python optimize.py -c config/benchmark/davis_cv0.json -n 50 \
  --study_name davis_cv0_hpo \
  --storage sqlite:///davis_cv0_hpo.db \
  --use_wandb --wandb_project "ChemGLaM-Davis-HPO"
```

## オプション

### コマンドラインオプション一覧

```bash
python optimize.py \
  -c <config_file>              # ベース設定ファイル（必須）
  -n <n_trials>                 # 試行回数（デフォルト: 50）
  --study_name <name>           # Study名（デフォルト: chemglam_optimization）
  --storage <url>               # データベースURL（例: sqlite:///optuna.db）
  --use_wandb                   # Weights & Biasを使用
  --wandb_project <name>        # W&Bプロジェクト名（デフォルト: ChemGLaM-Optimization）
  --direction <minimize|maximize>  # 最適化方向（task_typeから自動検出）
  --pruner <median|percentile|hyperband>  # Pruningアルゴリズム（デフォルト: median）
  --sampler <tpe|random|grid>   # サンプリングアルゴリズム（デフォルト: tpe）
```

### Prunerの選択

```bash
# Median pruner（デフォルト、バランス型）
python optimize.py -c config.json -n 50 --pruner median

# Hyperband pruner（積極的な枝刈り）
python optimize.py -c config.json -n 100 --pruner hyperband

# Percentile pruner（保守的な枝刈り）
python optimize.py -c config.json -n 50 --pruner percentile
```

### Samplerの選択

```bash
# TPE sampler（デフォルト、ベイズ最適化）
python optimize.py -c config.json -n 50 --sampler tpe

# Random sampler（ランダムサーチ）
python optimize.py -c config.json -n 50 --sampler random

# Grid sampler（全組み合わせを系統的に探索）
python optimize.py -c config.json -n 288 --sampler grid
```

## 出力ファイル

### 各トライアルのログ

```
./logs/optuna/davis_cv0/trial_0/
├── best_checkpoint.ckpt      # 最良のモデルチェックポイント
└── metrics.csv                # Epoch毎のvalidation metrics
```

- `trial_0`, `trial_1`, ..., `trial_49` のように各試行ごとにディレクトリが作成されます

### 最適化結果

```
./logs/optuna/davis_cv0_hpo/
├── best_config.json           # 最良のハイパーパラメータ
└── optimization_history.csv   # 全試行の履歴
```

### 結果の確認

```bash
# 最良のハイパーパラメータを確認
cat logs/optuna/davis_cv0_hpo/best_config.json

# 全試行の履歴を確認
cat logs/optuna/davis_cv0_hpo/optimization_history.csv

# 特定のtrialのvalidation結果を確認
cat logs/optuna/davis_cv0/trial_0/metrics.csv
```

## 実行例

### 例1: Davis CV0データセットでの最適化

```bash
python optimize.py \
  -c config/benchmark/davis_cv0.json \
  -n 50 \
  --study_name davis_cv0_hpo \
  --storage sqlite:///davis_cv0_hpo.db \
  --use_wandb \
  --wandb_project "ChemGLaM-Davis-HPO"
```

### 例2: PDBbindデータセットでの最適化

```bash
python optimize.py \
  -c local_config/prot_lang/pdbbind_cv0_esm150m.json \
  -n 100 \
  --study_name pdbbind_cv0_hpo \
  --storage sqlite:///pdbbind_cv0_hpo.db \
  --pruner hyperband \
  --use_wandb \
  --wandb_project "ChemGLaM-PDBbind-HPO"
```

### 例3: WandBを使わない場合

```bash
python optimize.py \
  -c config/benchmark/davis_cv0.json \
  -n 50 \
  --study_name davis_cv0_hpo \
  --storage sqlite:///davis_cv0_hpo.db
```

## 最適化の目的関数

最適化の方向（minimize/maximize）は`task_type`から自動的に判定されます：

- **Classification**: `(ROC-AUC + PR-AUC) / 2` を最大化
- **Regression**: `avg_val_loss` を最小化

### Checkpoint保存の基準

各トライアル内で最良のepochを保存する基準：

- **Classification**: `val_avg_auc` = `(ROC-AUC + PR-AUC) / 2` を最大化
- **Regression**: `avg_val_loss` を最小化

## 最適化されたパラメータでの学習

最適化が完了したら、`best_config.json`を使って学習を実行できます：

```bash
# 最良のパラメータで学習
python train.py -c logs/optuna/davis_cv0_hpo/best_config.json
```

## Tips

### 1. 探索効率について

- **50試行**: 288通りの探索空間の約17%をカバー。TPE samplerにより効率的に探索
- **100試行**: より確実に良い解を見つけたい場合（約35%カバー）
- **288試行（Grid sampler）**: 全組み合わせを試す場合

### 2. 並列実行

同じstorageを指定すれば、複数のプロセスで並列実行できます：

```bash
# ターミナル1
python optimize.py -c config.json -n 25 --storage sqlite:///hpo.db --study_name hpo

# ターミナル2（同時実行）
python optimize.py -c config.json -n 25 --storage sqlite:///hpo.db --study_name hpo
```

### 3. Gradient Accumulation

batch_size > 16を指定すると、自動的にgradient accumulationが適用されます：

- batch_size=32 → 実際のbatch_size=16, accumulate_grad_batches=2
- batch_size=64 → 実際のbatch_size=16, accumulate_grad_batches=4

これにより、メモリ効率を保ちながら大きな実効batch sizeでの学習が可能です。

### 4. Early Stopping (Pruning)

各trialは以下をモニタリングし、有望でない試行は早期に打ち切られます：

- **Classification**: `val_avg_auc` = `(ROC-AUC + PR-AUC) / 2`
- **Regression**: `avg_val_loss`

Pruningの設定（MedianPruner）：
- 最初の5 trials: pruningされない（比較対象を集める）
- 最初の3 epochs: pruningされない（学習初期の不安定さを考慮）

これにより、より多くの有望な組み合わせを試すことができます。

## トラブルシューティング

### エラー: "Study already exists"

既存のstudyを再利用する場合は問題ありません。新しいstudyを作成したい場合は、`--study_name`を変更してください。

### メモリ不足エラー

`batch_size`を小さくするか、`num_gpus`を増やしてください。

### WandBログインエラー

```bash
wandb login
```

を実行してAPIキーを設定してください。

## 参考リンク

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases](https://wandb.ai/)
