from pathlib import Path

import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import load_graphs

from src.datamodules.pde import GraphDatasetSequence, custom_collate


class BlackSeaSequence(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 7,
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.train_graphs, _ = load_graphs(str(self.data_dir / "train.bin"))
        self.val_graphs, _ = load_graphs(str(self.data_dir / "val.bin"))
        self.test_graphs, _ = load_graphs(str(self.data_dir / "test.bin"))

    def setup(self, stage: Optional[str] = None):
        train_dataset = GraphDatasetSequence(self.train_graphs, self.sequence_length)
        val_dataset = GraphDatasetSequence(self.val_graphs, self.sequence_length)
        test_dataset = GraphDatasetSequence(self.test_graphs, self.sequence_length)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )

    def train_dataloader(self, *args, **kwargs):
        return self.train_loader

    def val_dataloader(self, *args, **kwargs):
        return self.val_loader

    def test_dataloader(self, *args, **kwargs):
        return self.test_loader


if __name__ == "__main__":
    datamodule = BlackSeaSequence(data_dir="data/black_sea/graphs")
    print(len(datamodule.train_loader))
    print(len(datamodule.val_loader))
    print(len(datamodule.test_loader))
