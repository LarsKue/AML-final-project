import os

from datamodule import ResponseDataModule
from policy_tracker import PolicyTracker
from trainer import trainer

import subprocess as sub


def tensorboard():
    sub.run(["tensorboard", "--logdir", "lightning_logs"])


def main():
    dm = ResponseDataModule()
    pt = PolicyTracker()
    trainer.fit(pt, dm)
    tensorboard()


if __name__ == "__main__":
    main()
