import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import ResponseDataModule

import subprocess as sub

import pathlib
import re

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np


class PolicyTrackerMDN(pl.LightningModule):
    def __init__(self):
        super(PolicyTrackerMDN, self).__init__()

        self.num_gaussians = 1

        self.n_policies = 46
        self.n_other = 2

        self.example_input_array = torch.zeros(1, self.n_policies + self.n_other)

        self.base = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.PReLU(),
        )

        self.means = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )
        self.variances = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )
        self.weights = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_gaussians)
        )

        # perform forward pass to initialize lazy modules
        with torch.no_grad():
            self(self.example_input_array)

        # initialize weights
        for module in self.base.modules():
            if type(module) is nn.Linear:
                data = module.weight.data
                nn.init.normal_(data, 0, 2 / data.numel())

        def loss_fn(target, mu, sigma_sqr, pi):
            # pi, sigma, mu: (batch_size, num_gaussians)
            # print()
            # print("target", target.shape)
            # print("pi", pi.shape)
            # print("sigma_sqr", sigma_sqr.shape)
            # print("mu", mu.shape)

            exponents = -(target.expand_as(mu) - mu) ** 2 / (2 * sigma_sqr)
            max_exponent = torch.max(exponents, dim=1).values
            # print(exponents)
            # print(max_exponent)
            # print(exponents.shape)
            # print(max_exponent.shape)
            # print(exponents - max_exponent.unsqueeze(1).expand_as(exponents))

            gaussian_prob = torch.exp(exponents - max_exponent.unsqueeze(1).expand_as(exponents)) / torch.sqrt(
                2 * math.pi * sigma_sqr)

            # print("gaussian_prob", gaussian_prob.shape)
            # print("\n")
            # print(target)
            # print(mu)
            # print(sigma_sqr)
            # print(pi)
            # print(gaussian_prob)

            prob = pi * gaussian_prob
            prob[torch.isinf(gaussian_prob) & (pi < 1e-10)] = 0.0
            # print("prob", prob.shape)
            negative_log_likelihood = -torch.log(torch.sum(prob, dim=1)) - max_exponent
            # print("negative_log_likelihood", negative_log_likelihood.shape)

            return torch.mean(negative_log_likelihood)

        self.loss = loss_fn

    def forward(self, x):
        z = self.base(x)

        mu = self.means(z)
        sigma = F.elu(self.variances(z)) + 1
        pi = F.softmax(self.weights(z), dim=1)

        return mu, sigma, pi

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma_sqr, pi = self(x)

        loss = self.loss(y, mu, sigma_sqr, pi)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma_sqr, pi = self(x)

        loss = self.loss(y, mu, sigma_sqr, pi)

        self.log("val_loss", loss)

        return loss


def tensorboard():
    """
    Create a detached process for tensorboard
    """
    args = ["tensorboard", "--logdir", "lightning_logs"]

    process = sub.Popen(
        args, shell=False, stdin=None, stdout=None, stderr=None,
        close_fds=True
    )

    return process


def latest_version(path):
    """
    Returns latest model version as integer
    """
    # unsorted
    versions = list(str(v.stem) for v in path.glob("version_*"))
    # get version numbers as integer
    versions = [re.match(r"version_(\d+)", version) for version in versions]
    versions = [int(match.group(1)) for match in versions]

    return max(versions)


def latest_checkpoint(version=None):
    """
    Returns latest checkpoint path for given version (default: latest) as string
    """
    path = pathlib.Path("lightning_logs")
    if version is None:
        version = latest_version(path)
    path = path / pathlib.Path(f"version_{version}/checkpoints/")

    checkpoints = list(str(cp.stem) for cp in path.glob("*.ckpt"))

    # find epoch and step numbers
    checkpoints = [re.match(r"epoch=(\d+)-step=(\d+)", cp) for cp in checkpoints]

    # assign steps to epochs
    epoch_steps = {}
    for match in checkpoints:
        epoch = match.group(1)
        step = match.group(2)

        if epoch not in epoch_steps:
            epoch_steps[epoch] = []

        epoch_steps[epoch].append(step)

    # find highest epoch and step
    max_epoch = max(epoch_steps.keys())
    max_step = max(epoch_steps[max_epoch])

    # reconstruct path
    checkpoint = path / f"epoch={max_epoch}-step={max_step}.ckpt"

    return str(checkpoint)


def plot_country(model, df, country="Germany", randomize_policies=False):
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    x = torch.Tensor(x)
    y = torch.Tensor(y).unsqueeze(-1)

    if randomize_policies:
        # this is useful to check how much the model relies on this vs other features
        random_x = torch.randint(0, 1, size=(x.shape[0], model.n_policies)).to(torch.float64)
        x[:, :model.n_policies] = random_x

    mu, sigma_sqr, pi = model(x)

    max_component = torch.argmax(pi, dim=1)
    arange = torch.arange(len(max_component))

    mu = mu[arange, max_component].detach().cpu().numpy()
    sigma = torch.sqrt(sigma_sqr[arange, max_component]).detach().cpu().numpy()

    ax = plt.gca()
    ax.plot(np.arange(len(y)), y, label="Actual")

    ax.plot(np.arange(len(y)), mu, label="Predicted")
    ax.fill_between(np.arange(len(y)), mu - sigma, mu + sigma, color="C0", alpha=0.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()


def plot_countries(model, countries=("Germany",), randomize_policies=False):
    df = pd.read_csv("policies_onehot_full.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        axes.append(plt.subplot(nrows, ncols, i + 1))
        plot_country(model, df, country, randomize_policies=randomize_policies)

    # set all ylims equal
    ylims = []
    for ax in axes:
        ylims.extend(ax.get_ylim())

    ylims = [min(ylims), max(ylims)]
    for ax in axes:
        ax.set_ylim(ylims)

    plt.savefig("test.png")
    # plt.show()

    return ylims


def plot_country_heatmap(model, df, ylims, country="Germany", randomize_policies=False):
    df = df[df["country"] == country]

    df.pop("country")

    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    x = torch.Tensor(x)
    y = torch.Tensor(y).unsqueeze(-1)

    if randomize_policies:
        # this is useful to check how much the model relies on this vs other features
        random_x = torch.randint(0, 1, size=(x.shape[0], model.n_policies)).to(torch.float64)
        x[:, :model.n_policies] = random_x

    mu, sigma_sqr, pi = model(x)

    num_ys = 1000

    y_grid = torch.linspace(ylims[0], ylims[1], num_ys)
    y_grid = y_grid.repeat(mu.shape[0], 1)
    y_grid = y_grid.unsqueeze(2).repeat(1, 1, mu.shape[1])

    mu = mu.unsqueeze(1).repeat(1, num_ys, 1)
    sigma_sqr = sigma_sqr.unsqueeze(1).repeat(1, num_ys, 1)
    pi = pi.unsqueeze(1).repeat(1, num_ys, 1)

    exponents = -(y_grid - mu) ** 2 / (2 * sigma_sqr)
    gaussian_prob = torch.exp(exponents) / torch.sqrt(2 * math.pi * sigma_sqr)

    prob = pi * gaussian_prob
    prob = torch.sum(prob, dim=2)

    prob /= torch.max(prob)
    prob = prob.detach().cpu().numpy()

    ax = plt.gca()

    ax.plot(np.arange(len(y)), y, label="Actual")

    ax.imshow(prob.T, extent=[0, len(y), *ylims], origin="lower", aspect='auto', cmap='hot', interpolation='nearest')
    ax.autoscale(False)

    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()


def plot_countries_heatmap(model, ylims, countries=("Germany",), randomize_policies=False):
    df = pd.read_csv("policies_onehot_full.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    plt.figure(figsize=(6 * ncols + 1, 6 * nrows))

    axes = []
    for i, country in enumerate(countries):
        plt.subplot(nrows, ncols, i + 1)
        plot_country_heatmap(model, df, ylims, country, randomize_policies=randomize_policies)

    plt.savefig("test.png")
    # plt.show()

    return ylims


def plot_single_policy(model):
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols)

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)

        for j in range(6):
            policy = np.zeros(model.n_policies + model.n_other)
            policy[j] = 1

            x = np.tile(policy, (101, 1))
            x[:, -2] = 2 * i * np.ones(len(x))
            x[:, -1] = np.linspace(0, 1, 101)

            x = torch.Tensor(x)

            mu, sigma_sqr, pi = model(x)
            max_component = torch.argmax(pi, dim=1)
            arange = torch.arange(len(max_component))
            mu = mu[arange, max_component].detach().cpu().numpy()
            sigma = torch.sqrt(sigma_sqr[arange, max_component]).detach().cpu().numpy()

            ax = plt.gca()
            ax.plot(np.linspace(0, 1, 101), mu, label=j)
            ax.fill_between(np.linspace(0, 1, 101), mu - sigma, mu + sigma, alpha=0.2)
            # ax.set_xlabel("Vaccinations")
            # ax.set_ylabel("Delta R")
            ax.set_title(f"{2 * i} days")
            # ax.legend()

    # plt.show()


def plot_policies_vaccination(model, vaccination):
    policies = np.eye(model.n_policies)

    x = np.zeros((model.n_policies + 1, model.n_policies + 2))
    x[1:, :-2] = policies
    x[:, -1] = vaccination * np.ones(model.n_policies + 1)
    x = torch.Tensor(x)

    mu, sigma_sqr, pi = model(x)
    max_component = torch.argmax(pi, dim=1)
    arange = torch.arange(len(max_component))
    mu = mu[arange, max_component].detach().cpu().numpy()
    sigma = torch.sqrt(sigma_sqr[arange, max_component]).detach().cpu().numpy()

    plt.figure()
    plt.errorbar(np.arange(model.n_policies + 1), mu, yerr=sigma, fmt='.')

    xticks = [
        "no",

        "C1 1",
        "C1 2",
        "C1 3",

        "C2 1",
        "C2 2",
        "C2 3",

        "C3 1",
        "C3 2",

        "C4 1",
        "C4 2",
        "C4 3",
        "C4 4",

        "C5 1",
        "C5 2",

        "C6 1",
        "C6 2",
        "C6 3",

        "C7 1",
        "C7 2",

        "C8 1",
        "C8 2",
        "C8 3",
        "C8 4",

        "E1 1",
        "E1 2",

        "E2 1",
        "E2 2",

        "H1 1",
        "H1 2",

        "H2 1",
        "H2 2",
        "H2 3",

        "H3 1",
        "H3 2",

        "H6 1",
        "H6 2",
        "H6 3",
        "H6 4",

        "H7 1",
        "H7 2",
        "H7 3",
        "H7 4",
        "H7 5",

        "H8 1",
        "H8 2",
        "H8 3",
    ]
    plt.xticks(np.arange(model.n_policies + 1), xticks, rotation='vertical')

    # plt.show()


def main():
    dm = ResponseDataModule()
    pt = PolicyTrackerMDN()

    callbacks = [
        # save model with lowest validation loss
        pl.callbacks.ModelCheckpoint(monitor="val_loss"),
        # stop when validation loss stops decreasing
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ]

    logger = TensorBoardLogger(save_dir="lightning_logs", name="", default_hp_metric=False, log_graph=True)

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        gpus=1,
    )

    process = tensorboard()

    trainer.fit(pt, datamodule=dm)

    checkpoint = latest_checkpoint()

    pt = PolicyTrackerMDN.load_from_checkpoint(checkpoint)

    pt.eval()

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    ylims = plot_countries(model=pt, countries=countries, randomize_policies=True)
    ylims = plot_countries(model=pt, countries=countries, randomize_policies=False)

    plot_countries_heatmap(model=pt, ylims=ylims, countries=countries, randomize_policies=True)
    plot_countries_heatmap(model=pt, ylims=ylims, countries=countries, randomize_policies=False)

    plot_single_policy(model=pt)
    plot_policies_vaccination(model=pt, vaccination=0)
    plot_policies_vaccination(model=pt, vaccination=1)

    plt.show()

    print("Press Enter to terminate Tensorboard.")
    input()

    process.terminate()


if __name__ == "__main__":
    main()
