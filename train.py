# Fine-tuning ChemGLaM 
import argparse
import json
import torch
import lightning as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.model.chemglam import ChemGLaM
from src.data.datamodule import DTIDataModule
from src.utils.config import Config

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="./config/config_demo.json")
    args = args.parse_args()

    #json_fileからconfigを読み込む
    config = Config(args.config)
    model = ChemGLaM(config)
    datamodule = DTIDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="./logs",
        filename="best_checkpoint"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir="./logs",
        devices=config.num_gpus,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
