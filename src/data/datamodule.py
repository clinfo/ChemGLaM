import lightning as L
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torch
import transformers
from rdkit import Chem
import re
import os

from src.utils.config import Config

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name, tokenizer=None):
        df = df[['smiles', 'target_sequence', measure_name]]
        df = df.dropna()
        self.measure_name = measure_name
        df['canonical_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        df['replaced_sequence'] = df['target_sequence'].apply(lambda seq: " ".join(list(re.sub(r"[UZOB]", "X", seq))))
        df['replaced_sequence'] = df['replaced_sequence'].apply(lambda seq: seq[:2048])
        
        df_good = df.dropna(subset=['canonical_smiles', 'replaced_sequence'])  # TODO - Check why some rows are na
        
        len_new = len(df_good)
        print('Dropped {} invalid smiles and sequence'.format(len(df) - len_new))
        self.df = df_good
        self.df = self.df.reset_index(drop=True)
        print("Length of dataset:", len(self.df))

    def __getitem__(self, index):

        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        replaced_sequence = self.df.loc[index, 'replaced_sequence']
        measures = self.df.loc[index, self.measure_name]
        return canonical_smiles, replaced_sequence, measures
  
    def __len__(self):
        return len(self.df)

def get_dataset(dataset_path, measure_name):
    df = pd.read_csv(dataset_path)
    print("Length of dataset:", len(df))
    dataset = PropertyPredictionDataset(df,  measure_name)
    return dataset

class DTIDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache_dir = self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.protein_tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D', trust_remote_code=True)
        self.smi_tokenizer = transformers.AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)

    def tokenize_and_cache_dataset(self, dataset, split):
        cache_file = os.path.join(self.cache_dir, f"tokenized_{split}.pt")
        
        # キャッシュが存在するか確認
        if os.path.exists(cache_file):
            print(f"Loading tokenized {split} dataset from cache: {cache_file}")
            tokenized_data = torch.load(cache_file)
            return tokenized_data
        
        # データセット全体をトークナイズ
        smiles, sequences, measures = zip(*[dataset[i] for i in range(len(dataset))])
        smi_tokens = self.smi_tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
        protein_tokens = self.protein_tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest", return_tensors="pt")
        
        if self.hparams.task_type == "classification":
            tokenized_data = (
                torch.tensor(smi_tokens['input_ids']), 
                torch.tensor(smi_tokens['attention_mask']),
                protein_tokens['input_ids'],
                protein_tokens['attention_mask'], 
                torch.tensor(measures, dtype=torch.long)
            )
        elif self.hparams.task_type == "regression":            
            tokenized_data = (
                torch.tensor(smi_tokens['input_ids']), 
                torch.tensor(smi_tokens['attention_mask']),
                protein_tokens['input_ids'],
                protein_tokens['attention_mask'], 
                torch.tensor(measures, dtype=torch.float)
            )
        else:
            raise ValueError("Task type not recognized")
        
        # キャッシュに保存
        torch.save(tokenized_data, cache_file)
        print(f"Tokenized {split} dataset saved to cache: {cache_file}")
        
        return tokenized_data
    
    def setup(self, stage: str = None):
        # TODO implement for dataset creation
        self.dataset = get_dataset(self.config.data_root, self.config.dataset_path, self.config.measure_name)
        dataset_size = len(self.dataset)
        train_size = int(self.config.train_ratio * dataset_size)
        val_size = int(self.config.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        tokenized_dataset = self.tokenize_and_cache_dataset(self.dataset, "full")

        # split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(tokenized_dataset, [train_size, val_size, test_size])
    
    def collate(self, batch):
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=False)

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized