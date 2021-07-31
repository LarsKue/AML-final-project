
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

        ["C1_School closing",
        "C1_Flag",
        "C2_Workplace closing",
        "C2_Flag",
        "C3_Cancel public events",
        "C3_Flag",
        "C4_Restrictions on gatherings",
        "C4_Flag",
        "C5_Close public transport",
        "C5_Flag",
        "C6_Stay at home requirements",
        "C6_Flag",
        "C7_Restrictions on internal movement",
        "C7_Flag",
        "C8_International travel controls",
        "E1_Income support",
        "E1_Flag",
        "E2_Debt/contract relief",
        "E3_Fiscal measures",
        "E4_International support",
        "H1_Public information campaigns",
        "H1_Flag",
        "H2_Testing policy",
        "H3_Contact tracing",
        "H4_Emergency investment in healthcare",
        "H5_Investment in vaccines",
        "H6_Facial Coverings",
        "H6_Flag",
        "H7_Vaccination policy",
        "H7_Flag",
        "H8_Protection of elderly people",
        "H8_Flag"]


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
