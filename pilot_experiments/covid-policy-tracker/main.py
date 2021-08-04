import os

from datamodule import ResponseDataModule
from policy_tracker import PolicyTracker
from trainer import trainer

import subprocess as sub



import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


def tensorboard():
    sub.run(["tensorboard", "--logdir", "lightning_logs"])


def main():
    dm = ResponseDataModule()
    pt = PolicyTracker()
    trainer.fit(pt, dm)
    tensorboard()


def test():
    model = PolicyTracker.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=200-step=2612.ckpt")
    #model = PolicyTracker.load_from_checkpoint("./lightning_logs/version_1/checkpoints/epoch=199-step=382799.ckpt")


    df = pd.read_csv("policies_onehot_Germany.csv")
    y = df.pop("reproduction_rate").to_numpy()
    x = df.to_numpy()

    # drop index
    x = x[:, 1:]

    x = torch.Tensor(x)
    y = torch.Tensor(y).unsqueeze(-1)

    print(x.shape)
    print(y.shape)


    predicted = model(x)

    plt.figure()
    plt.plot(y)
    plt.plot(predicted.detach().cpu().numpy())

    plt.show()

if __name__ == "__main__":
    #main()
    test()
