import lightning as L
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel, DataCollatorWithPadding
from rdkit import Chem
import re
import os
from tqdm import tqdm

from src.utils.config import Config


class DTIPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, protein_tokens, measure_name=None):
        self.df = df
        self.protein_tokens = protein_tokens
        self.measure_name = measure_name
        
    def __getitem__(self, index):
        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        target_id = self.df.loc[index, 'target_id']
        protein_token = self.protein_tokens[target_id]
        if self.measure_name is None and self.stage == 'predict':
            return canonical_smiles, protein_token, index
        else:
            measure = self.df.loc[index, self.measure_name]
            return canonical_smiles, protein_token, measure, index
    
    def __len__(self):
        return len(self.df)

class DTIDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache_dir = self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.protein_tokens_cache_file = os.path.join(self.cache_dir, f"tokenized_protein_tokens.pt")
        self.stage = None
        
        self.protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.protein_collator = DataCollatorWithPadding(tokenizer=self.protein_tokenizer)
        
        self.smi_tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
        
        self.load_csv(self.config.dataset_path) 
        
    def load_csv(self, path):
        df = pd.read_csv(path)
        df = df[['smiles', 'target_sequence', self.config.measure_name]]
        df = df.dropna(subset=['smiles', 'target_sequence', self.config.measure_name])
        # target_sequence に IDを付与する
        df['target_id'] = df['target_sequence'].factorize()[0]
        df['canonical_smiles'] = df['smiles'].apply(lambda smi: self.normalize_smiles(smi, canonical=True, isomeric=False))
        df['replaced_sequence'] = df['target_sequence'].apply(lambda seq: " ".join(list(re.sub(r"[UZOB]", "X", seq))))
        
        df_good = df.dropna(subset=['canonical_smiles', 'replaced_sequence'])  # TODO - Check why some rows are na
        
        len_new = len(df_good)
        self.df = df_good.reset_index(drop=True)
        print('Dropped {} invalid smiles and sequence'.format(len(self.df) - len_new))
        print("Length of dataset:", len(self.df))

         
    def prepare_data(self):
        # if not exists, tokenize and save cache
        if not os.path.exists(self.protein_tokens_cache_file):
            protein_tokens = {}
            df_unique_target = self.df.drop_duplicates(subset=['target_id']).reset_index(drop=True)
            for i in tqdm(range(len(df_unique_target))):
                target_id, target_sequence = df_unique_target.loc[i, 'target_id'], df_unique_target.loc[i, 'replaced_sequence']
                protein_token = self.protein_tokenizer(target_sequence, max_length=2050, truncation=True)
                protein_tokens[target_id] = protein_token
            
            self.protein_tokens = protein_tokens    
            torch.save(protein_tokens, self.protein_tokens_cache_file)

    def setup(self, stage: str = None):
        print(f"Loading tokenized dataset from cache")
        self.protein_tokens = torch.load(self.protein_tokens_cache_file)
        self.dataset = DTIPredictionDataset(self.df, self.protein_tokens, measure_name=self.config.measure_name)
        self.stage = stage
        if stage == 'fit':
            dataset_size = len(self.dataset)
            train_size = int(self.config.train_ratio * dataset_size)
            val_size = int(self.config.val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size
            # split dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        elif stage == 'test':
            self.test_dataset = self.dataset
        elif stage == 'predict':
            self.predict_dataset = self.dataset
        else:
            raise ValueError(f"Stage {stage} not recognized")
    
    
    def collate_fn(self, batch):
        if self.stage == 'predict':
            smiles, protein_token, index = zip(*batch)
            smi_tokens = self.smi_tokenizer(smiles, padding=True, add_special_tokens=True) # TODO: want to take out smiles from the tokenized cache (Calculation time may be almost the same...)
            protein_token = self.protein_collator(protein_token)
            batch = {"drug_ids": torch.tensor(smi_tokens['input_ids']),
                     "drug_mask": torch.tensor(smi_tokens['attention_mask']),
                     "target_ids": protein_token['input_ids'],
                     "target_mask": protein_token['attention_mask'],
                     "index": index}
            return batch
        else:
            smiles, protein_token, measures, index = zip(*batch)
            smi_tokens = self.smi_tokenizer(smiles, padding=True, add_special_tokens=True)
            protein_token = self.protein_collator(protein_token)
            batch = {"drug_ids": torch.tensor(smi_tokens['input_ids']),
                    "drug_mask": torch.tensor(smi_tokens['attention_mask']),
                    "target_ids": protein_token['input_ids'],
                    "target_mask": protein_token['attention_mask'],
                    "measures": torch.tensor(measures, dtype=torch.float).view(-1, self.config.num_classes),
                    "index": index}
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