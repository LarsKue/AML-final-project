import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import GitHubData

# TODO


class ResponseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        super(ResponseDataModule, self).__init__()

    def prepare_data(self):
        # download, tokenize, etc...
        pass

    def setup(self, stage=None):
        # split, transform, etc...
        pass

    def train_dataloader(self):
        return DataLoader(...)

    def val_dataloader(self):
        return DataLoader(...)