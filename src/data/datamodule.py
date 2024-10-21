import lightning as L
from torch.utils.data import random_split, DataLoader

from src.utils.config import Config


class DTIDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # TODO implement for tokenization

        # tokenize

        # save to disk

    def setup(self, stage: str = None):
        # TODO implement for dataset creation

        # load from disk
        self.dataset = None  # TODO: implement dataset creation

        # split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1])  # TODO: implement dataset splitting

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
