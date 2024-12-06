# Fine-tuning ChemGLaM 
import argparse
import json
import pandas as pd 
import torch
import lightning as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything

from src.model.chemglam import ChemGLaM
from src.data.datamodule import DTIDataModule
from src.utils.config import Config

import torch
torch.set_float32_matmul_precision('medium')

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="./config/config_demo.json")
    args = args.parse_args()

    #json_fileからconfigを読み込む
    config = Config(args.config)
    seed_everything(config.seed, workers=True)
    
    model = ChemGLaM.load_from_checkpoint(config.checkpoint_path, config=config)
    
    datamodule = DTIDataModule(config)
    
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    trainer = L.Trainer(
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir=f"./logs/{config.experiment_name}",
        devices=1,
        )
    
    result = trainer.predict(model, datamodule)
    
    predictions = []
    attention_weights = []
    for res in result:
        predictions.append(res[0])
        if config.save_attention_weight:
            weights = res[1]
            for i in range(weights.size(0)):
                attention_weights.append(weights[i, :, :])
                
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    
    predictions = torch.cat(predictions, dim=0)
    predictions = sigmoid(predictions)
    
    if config.save_attention_weight:
        torch.save(attention_weights, f"./logs/{config.experiment_name}/attention_weights.pt")
    
    df_pred = pd.DataFrame(predictions.cpu().numpy(), columns=["pred"])
    
    if config.target_columns is not None:
        df_pred[config.target_columns] = datamodule.dataset.df[config.target_columns].values
    df_pred.to_csv(f"./logs/{config.experiment_name}/prediction.csv", index=False)

if __name__ == "__main__":
    main()
