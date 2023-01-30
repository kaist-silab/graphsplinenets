import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import load_graphs


class Gaussian2D(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.graph_list = load_graphs(self.data_dir)[0]
        self.graph_list_len = len(self.graph_list)
        self.tv_split = int(0.8 * self.graph_list_len)
        self.vt_split = int(0.9 * self.graph_list_len)

    def setup(self, stage: Optional[str] = None):
        train_dataset = GraphDataset(self.graph_list[: self.tv_split])
        val_dataset = GraphDataset(self.graph_list[self.tv_split : self.vt_split])
        test_dataset = GraphDataset(self.graph_list[self.vt_split :])
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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


class Gaussian2DOverfitting(LightningDataModule):
    """Gaussian2D dataset, batch is a sequence of status."""

    def __init__(
        self, data_dir: str, sequence_length: int, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.graph_list = load_graphs(self.data_dir)[0]
        self.graph_list_len = len(self.graph_list)
        self.tv_split = int(0.8 * self.graph_list_len)
        self.vt_split = int(0.9 * self.graph_list_len)

    def setup(self, stage: Optional[str] = None):
        train_dataset = GraphDatasetSequence(
            self.graph_list[: self.tv_split], self.sequence_length
        )
        val_dataset = GraphDatasetSequence(
            self.graph_list[self.tv_split : self.vt_split], self.sequence_length
        )
        test_dataset = GraphDatasetSequence(
            self.graph_list[self.vt_split :], self.sequence_length
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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


class Gaussian2DSequence(LightningDataModule):
    """Gaussian2D dataset, batch is a sequence of status."""

    def __init__(
        self, data_dir: str, sequence_length: int, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.graph_list = load_graphs(self.data_dir)[0]
        self.graph_list_len = len(self.graph_list)
        self.tv_split = int(0.8 * self.graph_list_len)
        self.vt_split = int(0.9 * self.graph_list_len)

    def setup(self, stage: Optional[str] = None):
        train_dataset = GraphDatasetSequence(
            self.graph_list[: self.tv_split], self.sequence_length
        )
        val_dataset = GraphDatasetSequence(
            self.graph_list[self.tv_split : self.vt_split], self.sequence_length
        )
        test_dataset = GraphDatasetSequence(
            self.graph_list[self.vt_split :], self.sequence_length
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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


class Gaussian2DSpace(LightningDataModule):
    def __init__(
        self, data_dir1: str, data_dir2: str, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.graph_list1 = load_graphs(self.data_dir1)[0]
        self.graph_list2 = load_graphs(self.data_dir2)[0]
        self.graph_list_len = len(self.graph_list1)
        self.tv_split = int(0.8 * self.graph_list_len)
        self.vt_split = int(0.9 * self.graph_list_len)

    def setup(self, stage: Optional[str] = None):
        train_dataset = GraphDatasetSpace(
            self.graph_list1[: self.tv_split], self.graph_list2[: self.tv_split]
        )
        val_dataset = GraphDatasetSpace(
            self.graph_list1[self.tv_split : self.vt_split],
            self.graph_list2[self.tv_split : self.vt_split],
        )
        test_dataset = GraphDatasetSpace(
            self.graph_list1[self.vt_split :], self.graph_list2[self.vt_split :]
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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


class GraphDataset(Dataset):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        return self.graph_list[index]


class GraphDatasetSequence(Dataset):
    def __init__(self, graph_list, sequence_length):
        super().__init__()
        self.graph_list = graph_list
        self.sequence_length = sequence_length

    def __len__(self):
        return int(len(self.graph_list) / self.sequence_length)

    def __getitem__(self, index):
        return self.graph_list[
            index * self.sequence_length : (index + 1) * self.sequence_length
        ]


class GraphDatasetSpace(Dataset):
    def __init__(self, graph_list1, graph_list2):
        """
        Args:
            graph_list1: list of graphs with smaller mesh
            graph_list2: list of graphs with larger mesh
        """
        super().__init__()
        self.graph_list1 = graph_list1
        self.graph_list2 = graph_list2

    def __len__(self):
        return len(self.graph_list1) - 2

    def __getitem__(self, index):
        return self.graph_list1[index : index + 2], self.graph_list2[index : index + 2]


def custom_collate(dict):
    """
    Input:
        dict <list> [batch_size]: contains batch_size number of
            output graphs. E.g. batch_size = 2, the output would
            be [graph1, graph2].
    """
    return dict
