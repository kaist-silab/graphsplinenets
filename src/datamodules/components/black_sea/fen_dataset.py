import logging
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops as eo
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from more_itertools import chunked
from scipy.spatial import Delaunay
from torch.utils.data import DataLoader, Dataset, Sampler
from torchtyping import TensorType

from src.datamodules.components.black_sea.utils import DomainInfo, PeriodicEncoder, Standardizer, STBatch
from src.datamodules.components.black_sea.utils import CellPredicate, Domain
from src.datamodules.components.black_sea.utils import MeshConfig, sample_mesh

log = logging.getLogger("fen.black_sea")



MOTUCLIENT_INSTRUCTIONS = """
The Black Sea dataset needs to be downloaded with motuclient from the Copernicus
Marine Service. To get access, register an account on their website [1] and put
your credentials into your ~/.netrc as follows.

machine my.cmems-du.eu
  login <your-user>
  password <your-password>

[1] https://marine.copernicus.eu
""".strip()


@dataclass(frozen=True)
class BlackSeaBatchKey:
    ranges: list[tuple[Optional[int], Optional[int]]]
    context_steps: int


@dataclass
class BlackSeaStats:
    mean: TensorType["time", "feature"]
    std: TensorType["time", "feature"]

    def get_standardizer(self, t: TensorType["batch", "time"]):
        day = t.long() % 365
        mean = self.mean[day]
        std = self.std[day]
        return Standardizer(
            eo.repeat(mean, "b t f -> b t 1 f"), eo.repeat(std, "b t f -> b t 1 f")
        )


class BlackSeaDataset(Dataset):
    def __init__(
        self, file: Path, domain: Domain, domain_info: DomainInfo, stats: BlackSeaStats
    ):
        super().__init__()

        self.file = file
        self.domain = domain
        self.domain_info = domain_info
        self.stats = stats

        data = torch.load(self.file)
        self.t = torch.from_numpy(data["t"]).float()
        self.u = torch.from_numpy(data["u"]).float()

        self.time_encoder = PeriodicEncoder(
            base=torch.tensor(np.datetime64("2012-01-01").astype(float)).float(),
            period=torch.tensor(365.0).float(),
        )

    def __getitem__(self, key: BlackSeaBatchKey) -> STBatch:
        t = []
        u = []
        for start, end in key.ranges:
            slice_ = slice(start, end)
            t.append(self.t[slice_])
            u.append(self.u[slice_])

        t = torch.stack(t)
        return STBatch(
            domain=self.domain,
            domain_info=self.domain_info,
            t=t,
            u=torch.stack(u),
            context_steps=key.context_steps,
            standardizer=self.stats.get_standardizer(t),
            time_encoder=self.time_encoder,
        )


class BlackSeaSampler(Sampler):
    def __init__(
        self,
        dataset: BlackSeaDataset,
        target_steps: int,
        context_steps: int,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__(dataset)

        self.dataset = dataset
        self.target_steps = target_steps
        self.context_steps = context_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        seq_len = target_steps + context_steps
        self.indices = [
            (start, start + seq_len) for start in range(len(self.dataset.t) - seq_len)
        ]

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            indices = indices.copy()
            random.shuffle(indices)
        for chunk in chunked(indices, self.batch_size):
            yield BlackSeaBatchKey(ranges=chunk, context_steps=self.context_steps)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


class OutOfDomainPredicate(CellPredicate):
    """Filter out all mesh cells that include mostly out-of-domain points."""

    def __init__(self, tri: Delaunay, x: np.ndarray, in_domain: np.ndarray):
        """
        Arguments
        ---------
        tri
            A mesh defined over some points
        x
            A set of "trial points" to check the mesh cells against
        in_domain
            A mask that says which points in `x` are in-domain
        """

        vertices = tri.points[tri.simplices]

        a, b, c = np.split(vertices, 3, axis=-2)
        ab, bc, ca = b - a, c - b, a - c
        ax, bx, cx = x - a, x - b, x - c

        # A point is inside a triangle if all these cross-products have the same sign
        abx = np.cross(ab, ax)
        bcx = np.cross(bc, bx)
        cax = np.cross(ca, cx)
        inside = ((abx * bcx) >= 0) & ((bcx * cax) >= 0) & ((cax * abx) >= 0)

        n_in_domain_in_cell = np.logical_and(inside, in_domain).sum(axis=-1)
        n_out_of_domain_in_cell = np.logical_and(inside, ~in_domain).sum(axis=-1)
        self.filter = n_out_of_domain_in_cell > n_in_domain_in_cell

    def __call__(self, cell_idx, cell, boundary_faces):
        return self.filter[cell_idx]
