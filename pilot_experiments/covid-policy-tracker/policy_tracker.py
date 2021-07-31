
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



        responses, responses2, testing = data.fetch()

        responses2.load()
        testing.load()

        responses2 = responses2.to_df()
        testing = testing.to_df()

        lb = LabelBinarizer()
        responses2["L1"] = lb.fit_transform(responses2["Measure_L1"]).tolist()

        print(responses2["L1"])

        x = torch.Tensor(responses2["L1"].values)

        print(x.shape)

        # TODO: add policies
        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)

    def val_dataloader(self):
        _, responses2, testing = data.fetch()

        responses2.load()
        testing.load()

        responses2 = responses2.to_df()
        testing = testing.to_df()

        lb = LabelBinarizer()
        responses2["L1"] = lb.fit_transform(responses2["Measure_L1"]).tolist()

        print(responses2["L1"])

        # x = torch.Tensor(testing["new_cases_smoothed_per_million"].values)
        x = torch.Tensor(responses2["L1"].values)

        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)
