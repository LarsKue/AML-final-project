
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

        responses = responses.to_df(low_memory=False)
        testing = testing.to_df()

        columns = [
            "C1_School closing",
            "C2_Workplace closing",
            "C3_Cancel public events",
            "C4_Restrictions on gatherings",
            "C5_Close public transport",
            "C6_Stay at home requirements",
            "C7_Restrictions on internal movement",
            "C8_International travel controls",
            "E1_Income support",
            "E2_Debt/contract relief",
            "H1_Public information campaigns",
            "H2_Testing policy",
            "H3_Contact tracing",
            "H6_Facial Coverings",
            "H7_Vaccination policy",
            "H8_Protection of elderly people"
       ]

        responses = responses[columns]

        max_values = [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3]
        n = sum(max_values)

        responses_tensor = torch.zeros(len(responses), n)

        # TODO: optimize, or just save and load
        for i, row in responses.iterrows():
            oh = torch.zeros(n)
            offset = 0

            for j, index in enumerate(row):
                try:
                    oh[offset + int(index) - 1] = 1.0
                except ValueError:
                    # ignore NaNs
                    pass
                offset += max_values[j]

            responses_tensor[i] = oh

        print(responses_tensor.shape)
        print(responses_tensor)

        # TODO: add epidemic state (vaccination rate)
        x = responses_tensor
        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)

    def val_dataloader(self):
        self.train_dataloader()
        responses, _, testing = data.fetch()

        responses.load()
        testing.load()

        responses = responses.to_df(low_memory=False)
        testing = testing.to_df()

        x = None
        y = torch.Tensor(testing["reproduction_rate"].values)

        ds = TensorDataset(x, y)

        return DataLoader(ds)



# def to_one_hot(indices):
#     max_values = [3, 3, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 4, 5, 3]
#     offset = 0
#     oh = torch.zeros(sum(max_values))
#     for i, mv in zip(indices, max_values):
#         try:
#             oh[offset + int(i) - 1] = 1.0
#         except ValueError:
#             # ignore NaNs
#             pass
#         offset += mv
#
#     return oh
