
import pytorch_lightning as pl

import torch
import torch.nn as nn


class PolicyTrackerSingle(pl.LightningModule):
    def __init__(self):
        super(PolicyTrackerSingle, self).__init__()

        # placeholder mean R0 taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7751056/
        # could be country-dependent, this should definitely be improved
        self.R0 = 3.28

        self.n_policies = 46
        self.n_other = 2

        self.example_input_array = torch.zeros(1, self.n_policies + self.n_other)

        self.model = nn.Sequential(
            nn.LazyLinear(out_features=1024),
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
            nn.Linear(256, self.n_policies),
        )

        # perform forward pass to initialize lazy modules
        with torch.no_grad():
            self(self.example_input_array)

        # initialize weights
        for module in self.model.modules():
            if type(module) is nn.Linear:
                data = module.weight.data
                nn.init.normal_(data, 0, 2 / data.numel())

        self.loss = nn.MSELoss()

    # TODO: somehow this predicts 0, all the time
    def forward(self, x):
        # predict the effect p of single policies on the R value
        p = self.model(x)

        # we don't want the neural net to have to deal with signs, all values should be >=0
        p = torch.abs(p)

        # which policies were in effect
        which = x[..., :self.n_policies] >= 1e-9
        p[~which] = 1.0

        # take the product
        m = torch.prod(p, dim=-1)

        # scale with R0
        return self.R0 * m

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)

        self.log("val_loss", loss)

        return loss

