import pytorch_lightning as pl

from datamodule import ResponseDataModule
from policy_tracker import PolicyTracker

import subprocess as sub

import pathlib
import re

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np


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

    predicted = model(x).detach().cpu().numpy()

    ax = plt.gca()
    ax.plot(y, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.set_title(country)
    ax.legend()


def plot_countries(countries=("Germany",), randomize_policies=False):
    checkpoint = latest_checkpoint()
    model = PolicyTracker.load_from_checkpoint(checkpoint)

    df = pd.read_csv("policies_onehot_full.csv")

    nrows = int(round(np.sqrt(len(countries))))
    ncols = len(countries) // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 1, 6 * nrows))

    for i, country in enumerate(countries):
        plt.subplot(nrows, ncols, i + 1)
        plot_country(model, df, country, randomize_policies=randomize_policies)

    # set all ylims equal
    ylims = []
    for ax in axes.flat:
        ylims.extend(ax.get_ylim())

    ylims = [min(ylims), max(ylims)]
    for ax in axes.flat:
        ax.set_ylim(ylims)

    plt.savefig("test.png")
    plt.show()


def main():
    dm = ResponseDataModule()
    pt = PolicyTracker()

    # instead of the last checkpoint,
    # save the one with the smallest validation loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    callbacks = [checkpoint_callback, early_stopping]

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        gpus=1,
    )

    # trainer.fit(pt, datamodule=dm)

    process = tensorboard()

    countries = ("Germany", "Spain", "Italy", "Japan", "Australia", "Argentina")

    plot_countries(countries, randomize_policies=False)

    print("Press Enter to terminate Tensorboard.")
    input()

    process.terminate()


if __name__ == "__main__":
    main()
