
import torch
import torch.nn as nn
import pytorch_lightning as pl


class PolicyTracker(pl.LightningModule):
    def __init__(self):
        super(PolicyTracker, self).__init__()

        self.n_policies = 46
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

        return loss
