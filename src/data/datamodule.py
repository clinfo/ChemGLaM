import lightning as L
from torch.utils.data import random_split, Subset, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel, DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
import numpy as np
import re
import os
import json
from tqdm import tqdm
from typing import List

from src.utils.config import Config


class DTIPredictionDataset(torch.utils.data.Dataset):
    """
    protein embeddingsを事前に計算しておくためのDataset
    """
    def __init__(self, df, config, protein_tokens, protein_embeddings=None, target_columns: List[str] = None):
        self.df = df
        self.protein_tokens = protein_tokens
        self.protein_embeddings = protein_embeddings
        self.target_columns = target_columns
        self.target_values = self.df[target_columns].values
        self.config = config
        
    def __getitem__(self, index):
        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        target_id = self.df.loc[index, 'target_id']
        protein_token = self.protein_tokens[target_id]
        if self.config.featurization_type == "embedding":
            protein_embedding = self.protein_embeddings[target_id]
        else:
            protein_embedding = None
            
        if self.target_columns is None and self.stage == 'predict':
            return canonical_smiles, protein_token, protein_embedding, index
        else:
            measure = self.target_values[index]
            return canonical_smiles, protein_token, protein_embedding, measure, index
    
    def __len__(self):
        return len(self.df)
    

class DTIDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache_dir = f"cache/{self.config.experiment_name}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.protein_token_cache_file = os.path.join(self.cache_dir, f"protein_tokens.pt")
        self.protein_embedding_cache_file = os.path.join(self.cache_dir, f"protein_embeddings.pt")
        self.stage = None
        
        self.protein_tokenizer = AutoTokenizer.from_pretrained(self.config.protein_model_name, trust_remote_code=True)
        self.protein_collator = DataCollatorWithPadding(tokenizer=self.protein_tokenizer)
        if config.featurization_type == "embedding":
            self.protein_model = EsmModel.from_pretrained(self.config.protein_model_name, trust_remote_code=True)
        
        self.smi_tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
        
        self.load_csv(self.config.dataset_csv_path) 
        
    def load_csv(self, path):
        df = pd.read_csv(path)
        df = df[['smiles', 'target_sequence', *self.config.target_columns]]
        df = df.dropna(subset=['smiles', 'target_sequence', *self.config.target_columns])
        # target_sequence に IDを付与する
        df['target_id'] = df['target_sequence'].factorize()[0]
        df['canonical_smiles'] = df['smiles'].apply(lambda smi: self.normalize_smiles(smi, canonical=True, isomeric=False))
        df['replaced_sequence'] = df['target_sequence'].apply(lambda seq: " ".join(list(re.sub(r"[UZOB]", "X", seq))))
        
        df_good = df.dropna(subset=['canonical_smiles', 'replaced_sequence'])
        
        len_new = len(df_good)
        self.df = df_good.reset_index(drop=True)
        print('Dropped {} invalid smiles and sequence'.format(len(self.df) - len_new))
        print("Length of dataset:", len(self.df))

         
    def prepare_data(self):        
        # if not exists, tokenize and save cache
        if not os.path.exists(self.protein_token_cache_file):
            protein_tokens = {}
            if self.config.featurization_type == "embedding":    
                protein_embeddings = {}
            df_unique_target = self.df.drop_duplicates(subset=['target_id']).reset_index(drop=True)
            for i in tqdm(range(len(df_unique_target))):
                target_id, target_sequence = df_unique_target.loc[i, 'target_id'], df_unique_target.loc[i, 'replaced_sequence']
                protein_token = self.protein_tokenizer(target_sequence, max_length=2050, truncation=True)
                if self.config.featurization_type == "embedding":
                    with torch.no_grad():
                        input_ids = protein_token["input_ids"]
                        attention_mask = protein_token["attention_mask"]
                        input_ids = torch.tensor(input_ids).view(1, -1).to("cuda")
                        attention_mask = torch.tensor(attention_mask).view(1, -1).to("cuda")
                        self.protein_model.to('cuda')
                        embedding = self.protein_model(input_ids=input_ids, 
                                                       attention_mask=attention_mask).last_hidden_state[0, :, :].detach().cpu()
                        input_ids.to("cpu")
                        attention_mask.to("cpu")
                        protein_embeddings[target_id] = embedding
                protein_tokens[target_id] = protein_token
            self.protein_tokens = protein_tokens
            torch.save(protein_tokens, self.protein_token_cache_file)
            if self.config.featurization_type == "embedding": 
                self.protein_embeddings = protein_embeddings   
                torch.save(protein_embeddings, self.protein_embedding_cache_file)

    def setup(self, stage: str = None):
        print(f"Loading tokenized dataset from cache")
        self.protein_tokens = torch.load(self.protein_token_cache_file)
        if self.config.featurization_type == "embedding":
            self.protein_embeddings = torch.load(self.protein_embedding_cache_file)
        else:
            self.protein_embeddings = None
            
        self.dataset = DTIPredictionDataset(self.df, 
                                            self.config, 
                                            self.protein_tokens, 
                                            self.protein_embeddings, 
                                            target_columns=self.config.target_columns)
        self.stage = stage
        if stage == 'fit':
            if self.config.split_json_path is not None:
                print(f"Loading split from {self.config.split_json_path}")
                with open(self.config.split_json_path, 'r') as f:
                    split_json = json.load(f)
                train_indices = split_json['train']
                valid_indices = split_json['valid']
                test_indices = split_json['test']
                # split dataset
                self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
                self.val_dataset = torch.utils.data.Subset(self.dataset, valid_indices)
                self.test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
            else:  
                dataset_size = len(self.dataset)
                train_size = int(self.config.train_ratio * dataset_size)
                val_size = int(self.config.val_ratio * dataset_size)
                test_size = dataset_size - train_size - val_size
                # split dataset
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    random_split(self.dataset, [train_size, val_size, test_size])
        elif stage == 'test':
            self.test_dataset = self.dataset
        elif stage == 'predict':
            self.predict_dataset = self.dataset
        else:
            raise ValueError(f"Stage {stage} not recognized")
    
    
    def collate_fn(self, batch):
        if self.stage == 'predict':
            smiles, protein_token, protein_embedding, index = zip(*batch)
        else:
            smiles, protein_token, protein_embedding, measures, index = zip(*batch)
            
        smi_tokens = self.smi_tokenizer(smiles, padding=True, add_special_tokens=True) # TODO: want to take out smiles from the tokenized cache (Calculation time may be almost the same...)
        protein_token = self.protein_collator(protein_token)
        
        batch = {"drug_ids": torch.tensor(smi_tokens['input_ids']),
                    "drug_mask": torch.tensor(smi_tokens['attention_mask']),
                    "target_ids": protein_token['input_ids'],
                    "target_mask": protein_token['attention_mask'],
                    "index": index}
        
        if self.stage != "predict":
            measures = np.array(measures)
            batch["measures"] = torch.tensor(measures, dtype=torch.float).view(-1, self.config.num_classes)
        
        if self.config.featurization_type == "embedding":
            protein_embedding = pad_sequence(list(protein_embedding), batch_first=True, padding_value=0)      
            batch["target_embedding"] = protein_embedding
        
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        batch_size=self.config.batch_size,
                        num_workers=self.config.num_workers,
                        collate_fn=self.collate_fn,
                        shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)
        
    def normalize_smiles(self, smi, canonical, isomeric=False):
        try:
            normalized = Chem.MolToSmiles(
                Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
            )
        except:
            normalized = None
        return normalized