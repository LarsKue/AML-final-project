from datamodule import ResponseDataModule
from policy_tracker import PolicyTracker
from trainer import trainer

import subprocess as sub

import pathlib

import pandas as pd
import torch
import matplotlib.pyplot as plt


def tensorboard():
    sub.run(["tensorboard", "--logdir", "lightning_logs"])


def latest_version(path):
    # ["version_1", "version_2", ..., "version_n", "version_0"]
    versions = list(str(v.stem) for v in path.glob("version_*"))
    # "version_n"
    version_str = sorted(versions)[-1]
    # n
    version = int(version_str.split("_")[1])

    return version


def latest_checkpoint(version=None):
    path = pathlib.Path("lightning_logs")
    if version is None:
        version = latest_version(path)
    path = path / pathlib.Path(f"version_{version}/checkpoints/")

    checkpoints = list(str(cp) for cp in path.glob("*.ckpt"))
    checkpoint = sorted(checkpoints)[-1]

    return checkpoint


def main():
    dm = ResponseDataModule()
    pt = PolicyTracker()
    trainer.fit(pt, dm)
    tensorboard()


def test():
    checkpoint = latest_checkpoint()
    model = PolicyTracker.load_from_checkpoint(checkpoint)

    df = pd.read_csv("policies_onehot_Germany.csv")
    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    x = torch.Tensor(x)

    random_policies = torch.randint(low=0, high=1, size=(len(x), 46)).to(torch.float64)
    # x[:, 0:46] = random_policies

    y = torch.Tensor(y).unsqueeze(-1)

    print(x.shape)
    print(y.shape)

    predicted = model(x)

    plt.figure()
    plt.plot(y)
    plt.plot(predicted.detach().cpu().numpy())

    plt.show()


if __name__ == "__main__":
    # main()
    test()
