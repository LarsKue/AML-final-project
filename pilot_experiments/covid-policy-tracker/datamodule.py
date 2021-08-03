
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd


class ResponseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        # save the dataframe, in case we need it
        self.df = None

        # full dataset
        self.ds = None
        # split datasets
        self.train_ds = None
        self.val_ds = None
        super(ResponseDataModule, self).__init__()

    def prepare_data(self):
        # load from csv
        df = pd.read_csv("policies_onehot.csv")
        y = df.pop("reproduction_rate").to_numpy()
        x = df.to_numpy()

        # drop index
        x = x[:, 1:]

        x = torch.Tensor(x)
        y = torch.Tensor(y).unsqueeze(-1)

        self.df = df
        self.ds = TensorDataset(x, y)

    def setup(self, stage=None):
        # perform random split with fractional lengths
        train = int(0.8 * len(self.ds))
        val = len(self.ds) - train

        self.train_ds, self.val_ds = random_split(self.ds, [train, val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, num_workers=3, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=3, batch_size=self.batch_size)
