# Hyperparameter Optimization for ChemGLaM
import argparse
import json
import os
import torch
import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
from datetime import timedelta

from chemglam.model.chemglam import ChemGLaM
from chemglam.data.datamodule import DTIDataModule
from chemglam.utils.config import Config

torch.set_float32_matmul_precision('medium')


def objective(trial, base_config, args):
    """Optuna objective function for hyperparameter optimization."""

    # ハイパーパラメータの探索空間を定義
    config_dict = base_config.to_dict().copy()

    # 学習率 (離散値から選択)
    config_dict["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])

    # Dropout (離散値から選択)
    config_dict["dropout"] = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    # MLP layers
    config_dict["mlp_num_layers"] = trial.suggest_int("mlp_num_layers", 1, 4)

    # Batch size and gradient accumulation
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    config_dict["batch_size"] = min(batch_size, 16)  # 実際のbatch sizeは最大16

    # batch_size > 16の場合はgradient accumulationを使用
    accumulate_grad_batches = max(1, batch_size // 16)

    # Experiment name with trial number
    config_dict["experiment_name"] = f"{base_config.experiment_name}/trial_{trial.number}"
    config_dict["cache_dir"] = base_config.cache_dir

    # Print selected hyperparameters for this trial
    print("\n" + "="*60)
    print(f"Trial {trial.number}: Selected Hyperparameters")
    print("="*60)
    print(f"  learning_rate:    {config_dict['learning_rate']:.2e}")
    print(f"  dropout:          {config_dict['dropout']:.2f}")
    print(f"  mlp_num_layers:   {config_dict['mlp_num_layers']}")
    print(f"  batch_size:       {batch_size} (actual: {config_dict['batch_size']}, accumulate: {accumulate_grad_batches})")
    print("="*60 + "\n")

    # Create config object from dict
    class DictConfig:
        def __init__(self, d):
            self.__dict__ = d

        def __getitem__(self, key):
            return self.__dict__[key]

        def __getattr__(self, key):
            return self.__dict__[key]

        def to_dict(self):
            return self.__dict__

    config = DictConfig(config_dict)

    # Set seed
    seed_everything(config.seed, workers=True)

    # Initialize model and data
    model = ChemGLaM(config)
    datamodule = DTIDataModule(config)

    # Callbacks
    # Checkpoint保存: avg_val_lossで統一
    # Pruning/Objective: task_typeに応じた指標を使用
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"./logs/optuna/{config.experiment_name}",
        filename="best_checkpoint",
        enable_version_counter=False,
    )

    # Pruning and objective metric
    if config.task_type == "classification":
        monitor_metric = "val_avg_auc"
        monitor_mode = "max"
    else:
        monitor_metric = "avg_val_loss"
        monitor_mode = "min"

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitor_metric)

    # Logger (WandBのみ、CSVは使わない)
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=config.experiment_name,
            tags=["optuna", f"trial_{trial.number}"]
        )
        logger = wandb_logger
    else:
        logger = True  # デフォルトロガー

    # Multi-GPU strategy
    if config.num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            timeout=timedelta(minutes=30)
        )
    else:
        strategy = "auto"

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir=f"./logs/optuna/{config.experiment_name}",
        devices=config.num_gpus,
        strategy=strategy,
        callbacks=[checkpoint_callback, pruning_callback],
        logger=logger,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train
    trainer.fit(model, datamodule)

    # Return validation metric
    # Checkpointが保存されたエポックの値を返す
    # checkpoint_callbackのbest_model_scoreはavg_val_loss（checkpointの保存基準）
    best_loss = checkpoint_callback.best_model_score.item()

    if config.task_type == "classification":
        # Checkpointが保存されたエポックのval_avg_aucを取得
        # validation_results.csvから読み込む
        import pandas as pd
        csv_path = f"./logs/optuna/{config.experiment_name}/validation_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # avg_val_lossが最小の行を取得
            best_row = df.loc[df['avg_val_loss'].idxmin()]
            avg_auc = best_row['val_avg_auc']
            roc_auc = best_row['val_roc_auc']
            pr_auc = best_row['val_pr_auc']
            best_epoch = int(best_row['epoch'])
        else:
            # CSVがない場合は最後のエポックの値を使う
            avg_auc = trainer.callback_metrics.get("val_avg_auc", 0.0).item()
            roc_auc = trainer.callback_metrics.get("val_roc_auc", 0.0).item()
            pr_auc = trainer.callback_metrics.get("val_pr_auc", 0.0).item()
            best_epoch = trainer.current_epoch

        print("\n" + "="*60)
        print(f"Trial {trial.number}: Results")
        print("="*60)
        print(f"  Best Epoch:       {best_epoch}")
        print(f"  Best Loss:        {best_loss:.4f}")
        print(f"  ROC-AUC:          {roc_auc:.4f}")
        print(f"  PR-AUC:           {pr_auc:.4f}")
        print(f"  Avg AUC:          {avg_auc:.4f}")
        print("="*60 + "\n")

        return avg_auc
    else:
        # Regressionの場合はlossを返す（最小化）
        val_loss = best_loss

        print("\n" + "="*60)
        print(f"Trial {trial.number}: Results")
        print("="*60)
        print(f"  Validation Loss:  {val_loss:.4f}")
        print("="*60 + "\n")

        return val_loss


def update_best_results(trial, study, base_config, study_name):
    """Update best results after each trial if improved"""
    import shutil

    print(f"\n[DEBUG] update_best_results called for trial {trial.number}")

    # 成功したトライアルが存在しない場合は何もしない
    try:
        best_trial = study.best_trial
        print(f"[DEBUG] best_trial obtained: {best_trial.number}")
    except ValueError as e:
        # まだ成功したトライアルがない
        print(f"[DEBUG] ValueError: {e} - no successful trials yet")
        return

    # Use experiment_name consistently for all paths
    base_dir = f"./logs/optuna/{base_config.experiment_name}"
    trial_checkpoint = f"{base_dir}/trial_{trial.number}/best_checkpoint.ckpt"

    # Use same directory for best results (experiment_name, not study_name)
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nChecking trial {trial.number}...")
    print(f"  Looking for checkpoint: {trial_checkpoint}")
    print(f"  Checkpoint exists: {os.path.exists(trial_checkpoint)}")
    print(f"  Best trial so far: {best_trial.number} (value: {best_trial.value:.4f})")

    # Save optimization history after every trial
    df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, "optimization_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Updated optimization history: {csv_path}")

    # Check if this is the best trial so far
    if best_trial.number == trial.number:
        print(f"\n🎉 New best trial found! Trial {trial.number}")

        # Save best hyperparameters
        best_config = base_config.to_dict().copy()
        best_config.update(study.best_params)

        output_path = os.path.join(output_dir, "best_config.json")
        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=4)

        print(f"  ✓ Updated best config: {output_path}")

        # Copy best checkpoint to output directory
        if os.path.exists(trial_checkpoint):
            best_checkpoint_dest = os.path.join(output_dir, "best_checkpoint.ckpt")
            shutil.copy2(trial_checkpoint, best_checkpoint_dest)
            print(f"  ✓ Updated best checkpoint: {best_checkpoint_dest}")
        else:
            print(f"  ⚠ Warning: Checkpoint not found at {trial_checkpoint}")

    # 各トライアル終了後、そのトライアルのチェックポイントを削除
    if os.path.exists(trial_checkpoint):
        os.remove(trial_checkpoint)
        if best_trial.number == trial.number:
            print(f"  ✓ Cleaned up trial checkpoint (already copied)")
        else:
            print(f"  ✓ Cleaned up trial_{trial.number} checkpoint")

    print()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ChemGLaM")
    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Path to base config JSON file")
    parser.add_argument("-n", "--n_trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("-s", "--study_name", type=str, default="chemglam_optimization",
                       help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                       help="Optuna storage URL (e.g., sqlite:///optuna.db). If relative path, saved in logs/optuna/")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="ChemGLaM-Optimization",
                       help="W&B project name")
    parser.add_argument("--direction", type=str, default=None,
                       choices=["minimize", "maximize"],
                       help="Optimization direction (auto-detected from task_type if not specified)")
    parser.add_argument("--pruner", type=str, default="median",
                       choices=["median", "percentile", "hyperband"],
                       help="Pruning algorithm")
    parser.add_argument("--sampler", type=str, default="tpe",
                       choices=["tpe", "random", "grid"],
                       help="Sampling algorithm")

    args = parser.parse_args()

    # Load base config
    base_config = Config(args.config)

    # Process storage path: if relative, save in logs/optuna/
    if args.storage is not None and not args.storage.startswith('sqlite:////'):
        # Extract database filename from storage URL
        if args.storage.startswith('sqlite:///'):
            db_filename = args.storage.replace('sqlite:///', '')
        else:
            db_filename = args.storage

        # If not absolute path, save in logs/optuna/
        if not db_filename.startswith('/'):
            import os
            os.makedirs('./logs/optuna', exist_ok=True)
            args.storage = f"sqlite:///./logs/optuna/{db_filename}"
            print(f"Database will be saved to: ./logs/optuna/{db_filename}")

    # Auto-detect direction from task_type if not specified
    if args.direction is None:
        if base_config.task_type == "classification":
            args.direction = "maximize"
            print(f"Auto-detected direction: maximize (task_type={base_config.task_type})")
        else:
            args.direction = "minimize"
            print(f"Auto-detected direction: minimize (task_type={base_config.task_type})")

    # Create pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
    elif args.pruner == "percentile":
        pruner = optuna.pruners.PercentilePruner(percentile=25.0)
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()

    # Create sampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=base_config.seed)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=base_config.seed)
    elif args.sampler == "grid":
        sampler = optuna.samplers.GridSampler({
            "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
            "dropout": [0.1, 0.2, 0.3],
            "mlp_num_layers": [1, 2, 3, 4],
            "batch_size": [8, 16, 32, 64],
        })

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    # Create callback that runs after each trial completes
    def trial_callback(study, trial):
        """Callback function that runs after each trial completes"""
        update_best_results(trial, study, base_config, args.study_name)

    study.optimize(
        lambda trial: objective(trial, base_config, args),
        n_trials=args.n_trials,
        callbacks=[trial_callback],
        show_progress_bar=True,
    )

    # Print final results summary
    print("\n" + "="*50)
    print("Optimization finished!")
    print("="*50)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    output_dir = f"./logs/optuna/{args.study_name}"
    print(f"\nResults saved to: {output_dir}")
    print(f"  - best_config.json")
    print(f"  - best_checkpoint.ckpt")
    print(f"  - optimization_history.csv")


if __name__ == "__main__":
    main()
