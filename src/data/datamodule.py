import lightning as L
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from rdkit import Chem
import re
import os
from tqdm import tqdm

from src.utils.config import Config

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

class DTIPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name=None, tokenizer=None, stage=None):
        df = df[['smiles', 'target_sequence', measure_name]]
        df = df.dropna()
        # target_sequence に IDを付与する
        df['target_id'] = df['target_sequence'].factorize()[0]
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
        canonical_smiles = torch.tensor(canonical_smiles.values, dtype=torch.float32)
        target_id = self.df.loc[index, 'target_id']
        target_id = torch.tensor(target_id, dtype=torch.float32)
        if self.stage=='predict':
            return canonical_smiles, target_id
        else:
            measures = self.df.loc[index, self.measure_name]
            measures = torch.tensor(measures, dtype=torch.float32)
            return canonical_smiles, target_id, measures, index
  
    def __len__(self):
        return len(self.df)

class DTIDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache_dir = self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.protein_embedding_cache_file = os.path.join(self.cache_dir, f"tokenized_protein_embeddings.pt")
        self.protein_masks_cache_file = os.path.join(self.cache_dir, f"tokenized_protein_masks.pt")
        self.stage = None
    
    def prepare_data(self):
        # load tokenizer
        self.protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.smi_tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
        # load data
        df = pd.read_csv(self.config.dataset_path)
        # make dataset
        self.dataset = DTIPredictionDataset(df, measure_name=self.config.measure_name)
        
        # tokenize and save cache if not exists

        # check if cache exists
        if os.path.exists(self.protein_embedding_cache_file) and os.path.exists(self.protein_masks_cache_file):
            print(f"Loading tokenized dataset from cache")
            self.protein_embeddings = torch.load(self.protein_embeddings_cache_file)
            self.protein_masks = torch.load(self.protein_masks_cache_file)
        
        # if not exists, tokenize and save cache
        else:
            protein_embeddings = {}
            protein_masks = {}
            model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
            model.half().to("cuda")
            for i in tqdm(range(len(df))):
                target_id, target_sequence = df.loc[i, 'target_id'], df.loc[i, 'replaced_sequence']
                protein_token = self.protein_tokenizer(target_sequence, max_length=2050, padding="max_length", return_tensors="pt").to("cuda")
                output = model(**protein_token)
                protein_embeddings[target_id] = output.last_hidden_state[0].detach().cpu()
                protein_masks[target_id] = protein_token["attention_mask"][0].detach().cpu()
            
            self.protein_embeddings = protein_embeddings
            self.protein_masks = protein_masks
            torch.save(protein_embeddings, self.protein_embeddings_cache_file)
            torch.save(protein_masks, self.protein_masks_cache_file)
    
    def setup(self, stage: str = None):
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
            smiles, target_ids, index = zip(*batch)
            smi_tokens = self.smi_tokenizer(smiles, padding=True, add_special_tokens=True) # TODO: want to take out smiles from the tokenized cache (Calculation time may be almost the same...)
            protein_embeddings = [self.protein_embeddings[i] for i in target_ids]  # batch_size * (2050, 2560)
            protein_embeddings = torch.stack(protein_embeddings)  # (batch_size, 2050, 2560)
            protein_masks = [self.protein_masks[i] for i in target_ids]
            protein_masks = torch.stack(protein_masks)
            return (torch.tensor(smi_tokens['input_ids']),
                    torch.tensor(smi_tokens['attention_mask']),
                    protein_embeddings,
                    protein_masks,
                    torch.tensor(index)
                    )
        else:
            smiles, target_ids, measures, index = zip(*batch)
            smi_tokens = self.smi_tokenizer(smiles, padding=True, add_special_tokens=True)
            protein_embeddings = [self.protein_embeddings[i] for i in target_ids]  # batch_size * (2050, 2560)
            protein_embeddings = torch.stack(protein_embeddings)  # (batch_size, 2050, 2560)
            protein_masks = [self.protein_masks[i] for i in target_ids]
            protein_masks = torch.stack(protein_masks)
            return (torch.tensor(smi_tokens['input_ids']),
                    torch.tensor(smi_tokens['attention_mask']),
                    protein_embeddings,
                    protein_masks,
                    torch.tensor(measures, dtype=torch.long),
                    torch.tensor(index)
                    )


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
