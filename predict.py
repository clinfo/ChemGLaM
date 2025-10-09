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

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, default="./config/config_demo.json")
    args = args.parse_args()

    #json_fileからconfigを読み込む
    config = Config(args.config)
    print(config)
    seed_everything(config.seed, workers=True)

    # Load checkpoint and detect MLP structure
    checkpoint = torch.load(config.checkpoint_path, weights_only=False)
    state_dict = checkpoint['state_dict']

    # Detect which MLP structure is used in the checkpoint
    has_mlpnet_old = 'mlp_net.fc1.weight' in state_dict  # Very old structure
    has_mlpnet = 'mlp_net.layers.0.weight' in state_dict and 'mlp_net.layers.3.weight' in state_dict  # Old MLPNet
    has_mlpnetv2 = 'mlp_net.layers.0.layer_norm.weight' in state_dict  # New MLPNetv2

    print(f"Checkpoint MLP structure detection:")
    print(f"  - Very old MLPNet (fc1/fc2): {has_mlpnet_old}")
    print(f"  - Old MLPNet (layers): {has_mlpnet}")
    print(f"  - New MLPNetv2 (with LayerNorm): {has_mlpnetv2}")

    # Convert old MLPNet structure (fc1, fc2) to new structure (layers)
    keys_to_convert = [
        ('mlp_net.fc1.weight', 'mlp_net.layers.0.weight'),
        ('mlp_net.fc1.bias', 'mlp_net.layers.0.bias'),
        ('mlp_net.fc2.weight', 'mlp_net.layers.3.weight'),
        ('mlp_net.fc2.bias', 'mlp_net.layers.3.bias'),
    ]

    for old_key, new_key in keys_to_convert:
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    checkpoint['state_dict'] = state_dict

    # Save converted checkpoint temporarily
    temp_checkpoint_path = config.checkpoint_path.replace('.ckpt', '_converted.ckpt')
    torch.save(checkpoint, temp_checkpoint_path)

    model = ChemGLaM.load_from_checkpoint(temp_checkpoint_path, config=config)

    datamodule = DTIDataModule(config)

    # Set log directory based on wandb_project availability
    if 'wandb_project' in config:
        log_dir = f"./logs/{config.wandb_project}/{config.experiment_name}"
    else:
        log_dir = f"./logs/{config.experiment_name}"

    trainer = L.Trainer(
        enable_progress_bar=True,
        accelerator="gpu",
        gradient_clip_val=None,
        default_root_dir=log_dir,
        devices=1,
        )
    
    datamodule.prepare_data()
    if config.target_columns is not None:
        trainer.test(model, datamodule=datamodule, verbose=True)
    result = trainer.predict(model, datamodule=datamodule)
    
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
        torch.save(attention_weights, f"{log_dir}/attention_weights.pt")
    
    df_pred = pd.DataFrame(predictions.cpu().numpy(), columns=["pred"])
    if config.task_type == "classification" and config.evidential:
        df_pred["uncertainty"] = uncertainty.cpu().numpy()
    
    if config.target_columns is not None:
        df_pred[config.target_columns] = datamodule.dataset.df[config.target_columns].values
    df_pred.to_csv(f"{log_dir}/prediction.csv", index=False)

if __name__ == "__main__":
    main()
