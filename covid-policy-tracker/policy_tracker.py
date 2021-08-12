import torch
import torch.nn as nn
import pytorch_lightning as pl


class PolicyTracker(pl.LightningModule):
    def __init__(self):
        super(PolicyTracker, self).__init__()

        self.n_policies = 46
        self.n_other = 2

        self.model = nn.Sequential(
            nn.Linear(self.n_policies + self.n_other, 1024),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 1)
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

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
