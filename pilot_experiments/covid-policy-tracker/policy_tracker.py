
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer

from torch.utils.data import TensorDataset, DataLoader

import data


class PolicyTracker(pl.LightningModule):
    def __init__(self):
        super(PolicyTracker, self).__init__()

        self.n_policies = 8
        self.n_other = 1

        self.model = nn.Sequential(
            nn.Linear(self.n_policies + self.n_other, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)

        self.log("val_loss", loss)

    def train_dataloader(self):
        # TODO:
        #  - datasets need merging:
        #  - dataset needs to give the following information for a given index:
        #  - x: policy vector and epidemic state
        #  - y: R value
        #  - since dates differ in the two datasets, we need to merge them by assuming
        #  - that p, x, R are piecewise constant
        responses, _, testing = data.fetch()

        responses.load()
        testing.load()

        responses = responses.to_df()
        testing = testing.to_df()

        x = None
        # TODO: add policies
        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)

    def val_dataloader(self):
        responses, _, testing = data.fetch()

        responses.load()
        testing.load()

        responses = responses.to_df()
        testing = testing.to_df()

        x = None
        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)



def to_one_hot(indices):
    max_values = [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3]
    offset = 0
    oh = torch.zeros(sum(max_values))
    for i, mv in zip(indices, max_values):
        oh[offset + i] = 1.0
        offset += mv

    return oh
