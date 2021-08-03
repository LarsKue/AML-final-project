
from datamodule import ResponseDataModule
from policy_tracker import PolicyTracker
from trainer import trainer


def main():
    dm = ResponseDataModule()
    pt = PolicyTracker()
    trainer.fit(pt, dm)


if __name__ == "__main__":
    main()
