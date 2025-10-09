# Fine-tuning ChemGLaM
import argparse
import json
import torch
import lightning as L
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything

from chemglam.model.chemglam import ChemGLaM
from chemglam.data.datamodule import DTIDataModule
from chemglam.utils.config import Config

import torch
torch.set_float32_matmul_precision('medium')


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="./config/config_demo.json")
    args = args.parse_args()

    # json_fileからconfigを読み込む
    config = Config(args.config)
    seed_everything(config.seed, workers=True)

    model = ChemGLaM(config)
    datamodule = DTIDataModule(config)

    # Gradient accumulation: batch_size > 16の場合に使用
    accumulate_grad_batches = 1
    if hasattr(config, 'batch_size') and config.batch_size > 16:
        accumulate_grad_batches = config.batch_size // 16
        # 実際のbatch sizeを16に調整
        config.__dict__['batch_size'] = 16
        # データモジュールを再作成
        datamodule = DTIDataModule(config)

    # Get wandb project name from config, default to "ChemGLaM"
    wandb_project = getattr(config, 'wandb_project', 'ChemGLaM')

    # Log directory includes wandb project name only if explicitly specified
    if 'wandb_project' in config:
        log_dir = f"./logs/{wandb_project}/{config.experiment_name}"
    else:
        log_dir = f"./logs/{config.experiment_name}"

    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        mode="min",
        save_top_k=1,
        dirpath=log_dir,
        filename="best_checkpoint",
        enable_version_counter=False,
    )

    wandb_logger = WandbLogger(project=wandb_project, name=config.experiment_name)

    # Multi-GPU strategy
    if config.num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            timeout=timedelta(minutes=30)
        )
    else:
        strategy = "auto"

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir=log_dir,
        devices=config.num_gpus,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, datamodule, ckpt_path=config.checkpoint_path,)
    
if __name__ == "__main__":
    main()
