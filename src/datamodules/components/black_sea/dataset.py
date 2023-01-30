from rich.table import Table
from rich.console import Console
import datetime
import h5netcdf
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# from dl4dyn.managers.black_sea import BlackSeaManager


def fromtimestamp(ts):
    return str(datetime.datetime.fromtimestamp(ts))


class BlackSeaNormalizer:
    def __init__(self):
        raise NotImplementedError("Needs to be reimplemented")


class BlackSeaDataset(Dataset):
    """
    Black Sea Dataset downloaded from: https://resources.marine.copernicus.eu/product-detail/BLKSEA_MULTIYEAR_PHY_007_004/INFORMATION
    Reference for parts of the code, especially the downloader: https://github.com/martenlienen/finite-element-networks
    Data is downloaded from AWS S3 under data_path: note that it will keep S3's foldering structure, so you can use the same path for all datasets
    Args:
        files (list): List of files to load
        time_indexes (int): Number of time steps to load
        keys (list): List of keys to load, i.e. named channels
        data_path (str): Path to download files from
        context_steps (int): Number of time steps to use as context (past + current)
        target_steps (int): Number of time steps to use as target
        mask (torch.Tensor): Mask to apply to the data
        is_preprocessed (bool): If True, skip preprocessing
        transform (callable): Optional transform to be applied on a sample
        normalize (bool): If True, normalize the data
        f32 (bool): If True, convert to float32
    """

    def __init__(
        self,
        files=None,
        time_indexes=None,
        keys=None,
        data_path="",
        context_steps=1,
        target_steps=1,
        mask=None,
        is_preprocessed=False,
        transform=None,
        normalize=False,
        f32=True,
        download=True,
    ):

        # manager = BlackSeaManager(data_path, files, download=download)
        # if files is None: # provide a list of files to load
        #     files = manager.get_filepaths()
        self.data_files = [h5netcdf.File(file, "r") for file in files]

        if keys is None:  # by default, filename is the key
            keys = [str(Path(f.filename).stem) for f in self.data_files]
        self.keys = keys

        if time_indexes is None:  # load all times if indexes are not provided
            time_indexes = self.data_files[0]["time"].shape[0]

        if mask is None:
            mask = self._get_mask()
        self.mask = mask

        # Separate target and context steps (context is past + current)
        self.target_steps = target_steps
        self.context_steps = context_steps
        seq_len = target_steps + context_steps
        self.indices = [start_idx for start_idx in range(time_indexes - seq_len)]

        # Preprocessing
        self.is_preprocessed = is_preprocessed  # if True, skip the rest
        self.norm_fn = BlackSeaNormalizer() if normalize else None
        self.transform = transform
        self.f32 = f32

    def __getitem__(self, idx):
        # Load data from HDF5 file
        time_index = self.indices[idx]

        # Stack data from files
        # Shape: [time x chan x lat x lon]
        xs = []
        for idx_ in range(
            time_index, time_index + self.context_steps + self.target_steps
        ):
            x = np.stack(
                [
                    np.array(self.data_files[i][key][idx_]).squeeze()
                    for i, key in enumerate(self.keys)
                ],
                axis=0,
            )
            xs.append(x)
        xs = torch.tensor(np.array(xs))

        # Preprocessing
        if not self.is_preprocessed:
            if self.f32:
                xs = xs.float()
            if self.transform is not None:
                xs = self.transform(xs)
            if self.norm_fn is not None:
                xs = self.norm_fn(xs)

        inputs = xs[: -self.target_steps, ...]
        targets = xs[self.target_steps :, ...]
        return inputs, targets

    def _get_mask(self):
        # The mask is a binary tensor with 1s where the data is valid and 0s where the data is missing
        # In this case, NaN values are 1e20
        mask = np.absolute(self.data_files[0][self.keys[0]][0, 0] >= 1e19)
        return torch.tensor(mask)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        """Print rich summary of dataset"""
        table = Table(title="Black Sea Dataset")
        columns = ["Start date", "End date", "Keys", "Context steps", "Target steps"]
        for column in columns:
            table.add_column(column, justify="center", style="cyan")
        table.add_row(
            fromtimestamp(self.data_files[0]["time"][0]),
            fromtimestamp(self.data_files[0]["time"][-1]),
            "\n".join(str(d) for d in self.keys),
            str(self.context_steps),
            str(self.target_steps),
        )
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()


if __name__ == "__main__":
    dataset = BlackSeaDataset(data_path="data")
    print(dataset)
    x, y = next(iter(dataset))
    print(x.shape, y.shape)
