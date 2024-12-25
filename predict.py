# Fine-tuning ChemGLaM 
import argparse
import json
import pandas as pd 
import torch
import lightning as L
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


def predict(config):
    print(config)
    seed_everything(config.seed, workers=True)
    
    model = ChemGLaM.load_from_checkpoint(config.checkpoint_path, config=config)
    
    datamodule = DTIDataModule(config)
    
    trainer = L.Trainer(
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir=f"./logs/{config.experiment_name}",
        devices=1,
        )
    
    datamodule.prepare_data()
    if config.target_columns is not None:
        trainer.test(model, ckpt_path=config.checkpoint_path, datamodule=datamodule, verbose=True)
    result = trainer.predict(model, ckpt_path=config.checkpoint_path, datamodule=datamodule)
    
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
    
    if config.task_type == "classification" and not config.evidential:
        if config.num_classes == 1:
            predictions = sigmoid(predictions)
        else:
            predictions = F.softmax(predictions, dim=1)
    elif config.task_type == "classification" and config.evidential:
        evidence = F.softplus(predictions)
        alpha = evidence + 1 
        
        uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
        
        predictions = alpha / torch.sum(alpha, dim=1, keepdim=True)
        predictions = predictions[:, 1]
        
    if config.save_attention_weight:
        torch.save(attention_weights, f"./logs/{config.experiment_name}/attention_weights.pt")
    
    df_pred = pd.DataFrame(predictions.cpu().numpy(), columns=["pred"])
    if config.task_type == "classification" and config.evidential:
        df_pred["uncertainty"] = uncertainty.cpu().numpy()
    
    if config.target_columns is not None:
        df_pred[config.target_columns] = datamodule.dataset.df[config.target_columns].values
    df_pred.to_csv(f"./logs/{config.experiment_name}/prediction.csv", index=False)
    
    del model, datamodule, trainer, predictions, attention_weights, result
    torch.cuda.empty_cache()
    gc.collect()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="./config/config_demo.json")
    args.add_argument("-i", "--split_id", type=str, default=0)
    args = args.parse_args()
    
    with open(f"local_data/large_cpi/split_{args.split_id}/protein_entry.txt", "r") as f:
        uniprot_ids = f.read().splitlines()

    for uniprot_id in uniprot_ids: 
        config = Config(args.config)
        config.experiment_name = f"{config.experiment_name}/split_{args.split_id}/{uniprot_id}"
        config.cache_dir = f"{config.cache_dir}/split_{args.split_id}/{uniprot_id}"
        config.dataset_csv_path = f"local_data/large_cpi/split_{args.split_id}/{uniprot_id}.csv"
        predict(config)
        
        del config
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
