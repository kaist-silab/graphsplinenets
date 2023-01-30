import dgl
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn.functional import interpolate
from dgl.nn.pytorch.softmax import edge_softmax
from scipy.interpolate import griddata

from src.numerics.osc import osc1d, osc2d
from src.models.components.base import BaseGNN


class TimeSplineNets(BaseGNN):
    """SplineGraphNets with only time oriented collocation"""

    def __init__(self, sequence_length, dim, **kwargs):
        super().__init__(**kwargs)

        self.collection = []
        self.val_collection = []

        # Setup OSC1d
        self.sequence_length = sequence_length
        self.x = torch.tensor(
            np.linspace(0, 1, sequence_length, endpoint=True), dtype=torch.float32
        )
        self.dim = dim
        self.p = torch.tensor([0, 1])
        self.c = torch.tensor(
            np.linspace(0, 1, dim + 1, endpoint=True)[1:-1], dtype=torch.float32
        )

    def forward(self, g):
        # Encoder
        enc_n_feat = self.encoder(g.ndata["feat"])

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata["dist"])

        # Message Passing
        g.ndata["h"] = softmax_n_feat
        g.edata["h"] = g.edata["feat"]

        g.apply_edges(func=self.edge_update_func)
        g.pull(
            g.nodes(),
            message_func=dgl.function.copy_e("h", "m"),
            reduce_func=dgl.function.sum("m", "agg_m"),
        )
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata["h"], g.edata["dist"])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # (Important) Delete temp features
        _ = g.ndata.pop("h")
        _ = g.ndata.pop("agg_m")
        _ = g.ndata.pop("sum_m")
        _ = g.edata.pop("h")
        _ = g.edata.pop("w")
        _ = g.edata.pop("wh")

        return decode_node_feature

    def training_step(self, batch, batch_nb):
        """
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details
                please refer to data module class.
        """
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        out_graph_list = [inp_graph]
        gt = []
        node_feat = [inp_graph.ndata["feat"][:, 2]]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata["feat"][:, 2])
        gt = torch.stack(gt)

        # Sequence rollout
        # for i in range(int((self.sequence_length-1)/(self.dim-1))):
        step_len = int((self.sequence_length - 1) / self.dim)
        for i in range(0, self.sequence_length - step_len, step_len):
            inp_graph = batch[0][i].clone()
            # inp_graph.ndata['feat'][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)):  # For each node series
            y = node_feat[1:-1, i]
            b1 = node_feat[0, i]
            b2 = node_feat[-1, i]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            pred.append(f_(self.x))
        pred = torch.permute(torch.stack(pred), (1, 0))
        loss = nn.MSELoss()(gt, pred)
        self.collection.append(loss.item())
        return loss

    def training_epoch_end(self, epoch_output):
        self.log(
            "train/loss", sum(self.collection) / len(self.collection), prog_bar=True
        )
        self.collection = []

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details
                please refer to data module class.
        """
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        gt = []
        node_feat = [inp_graph.ndata["feat"][:, 2]]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata["feat"][:, 2])
        gt = torch.stack(gt)

        # Sequence rollout
        step_len = int((self.sequence_length - 1) / self.dim)
        for i in range(0, self.sequence_length - step_len, step_len):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata["feat"][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)):  # For each node series
            y = node_feat[1:-1, i]
            b1 = node_feat[0, i]
            b2 = node_feat[-1, i]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            pred.append(f_(self.x))
        pred = torch.permute(torch.stack(pred), (1, 0))
        loss = nn.MSELoss()(gt, pred)
        self.val_collection.append(loss.item())

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def validation_step_end(self, validation_step_output):
        self.log(
            "val/loss",
            sum(self.val_collection) / len(self.val_collection),
            prog_bar=False,
            batch_size=1,
        )
        self.val_collection = []


class SpaceSplineNets(BaseGNN):
    """SplineGraphNets with only space oriented collocation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.collection = []
        self.val_collection = []

    def forward(self, g):
        # Encoder
        enc_n_feat = self.encoder(g.ndata["feat"])

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata["dist"])

        # Message Passing
        g.ndata["h"] = softmax_n_feat
        g.edata["h"] = g.edata["feat"]

        g.apply_edges(func=self.edge_update_func)
        g.pull(
            g.nodes(),
            message_func=dgl.function.copy_e("h", "m"),
            reduce_func=dgl.function.sum("m", "agg_m"),
        )
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata["h"], g.edata["dist"])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # (Important) Delete temp features
        _ = g.ndata.pop("h")
        _ = g.ndata.pop("agg_m")
        _ = g.ndata.pop("sum_m")
        _ = g.edata.pop("h")
        _ = g.edata.pop("w")
        _ = g.edata.pop("wh")

        return decode_node_feature

    def training_step(self, batch, batch_nb):
        """
        This may require a special dataloader, and here I just
            directly use the cubic interpolation method
        Args:
            batch <list of list> [batch_size, 2]: mostly the batch_size
                will be set to 1, so the input are two graph.
                The first graph is the training graph, the second graph
                is the target one
        """
        inp_graph = batch[0][0][0]
        tar_feat = batch[0][1][1].ndata["feat"][:, 2]
        tar_feat = torch.reshape(tar_feat, (8, 8))

        out_feat = torch.squeeze(self(inp_graph))  # [64]

        # Pytorch interpolation method
        # Later can be replaced by the OSC method
        out_feat = torch.reshape(out_feat, (1, 1, 7, 7))
        out_feat_int = interpolate(
            out_feat, size=(8, 8), mode="bilinear", align_corners=False
        )
        out_feat_int = torch.squeeze(out_feat_int)

        # WARNING: this part doesn't work
        # Use interpolation
        # x = inp_graph.ndata['feat'][:, 0]
        # y = inp_graph.ndata['feat'][:, 1]
        # x, y = torch.meshgrid(x, y)
        # mesh = torch.transpose(torch.stack([x.flatten(), y.flatten()]), 1, 0)

        # x_ = torch.tensor(np.linspace(0, 1, 256, endpoint=True), dtype=torch.float32)
        # y_ = torch.tensor(np.linspace(0, 1, 256, endpoint=True), dtype=torch.float32)
        # x_ = tar_graph.ndata['feat'][:, 0]
        # y_ = tar_graph.ndata['feat'][:, 1]
        # x_, y_ = torch.meshgrid(x_, y_)
        # mesh_ = torch.transpose(torch.stack([x_.flatten(), y_.flatten()]), 1, 0)

        # sim_feat = griddata(mesh, out_feat, mesh_, method='cubic')

        # Calculate loss
        loss = nn.MSELoss()(out_feat_int, tar_feat)
        self.collection.append(loss.item())
        return loss

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def training_epoch_end(self, epoch_output):
        self.log(
            "train/loss", sum(self.collection) / len(self.collection), prog_bar=True
        )
        self.collection = []


class TimeSpaceSplineNets(BaseGNN):
    """Spline Graph Nets with both time and space interpolation."""

    def __init__(
        self,
        sequence_length,
        dim,
        part2d,
        col2d,
        nx,
        ny,
        nx_sim,
        ny_sim,
        dx,
        dy,
        r: int = 3,
        square_mesh=True,
        **kwargs
    ):
        """
        Args:
            sequence_length <int>: the length of the sequence of time oriented OSC method
            dim <int>: the dimension of time oriented OSC method
            part2d <torch.tensor>: the partitions in 2d space
            col2d <torch.tensor>: the collocation in 2d space
            nx <int>: the number of partitions in x direction
            ny <int>: the number of partitions in y direction
            nx_sim <int>: the number of simulation resolution in x direction
            ny_sim <int>: the number of simulation resolution in y direction
            dx <int>: the gap of partitions in x direction
            dy <int>: the gap of partitions in y direction
            r <int>: the dimension of space oriented OSC method
        """
        super().__init__(**kwargs)

        self.collection = []
        self.val_collection = []

        # Setup OSC1d
        self.sequence_length = sequence_length
        self.x = torch.tensor(
            np.linspace(0, 1, sequence_length, endpoint=True), dtype=torch.float32
        )
        self.dim = dim
        self.p = torch.tensor([0, 1])
        self.c = torch.tensor(
            np.linspace(0, 1, dim + 1, endpoint=True)[1:-1], dtype=torch.float32
        )

        # Setup OSC2d
        self.part2d = part2d
        self.col2d = col2d
        self.x_base = self._get_base(nx, nx_sim, dx)
        if not square_mesh:
            self.y_base = self._get_base(ny, ny_sim, dy)
        else:
            self.y_base = self.x_base
        self.base = torch.einsum("ij,kl->ikjl", self.x_base, self.y_base).to("cuda:0")

    def forward(self, g):
        # Encoder
        enc_n_feat = self.encoder(g.ndata["feat"])

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata["dist"])

        # Message Passing
        g.ndata["h"] = softmax_n_feat
        g.edata["h"] = g.edata["feat"]

        g.apply_edges(func=self.edge_update_func)
        g.pull(
            g.nodes(),
            message_func=dgl.function.copy_e("h", "m"),
            reduce_func=dgl.function.sum("m", "agg_m"),
        )
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata["h"], g.edata["dist"])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # (Important) Delete temp features
        _ = g.ndata.pop("h")
        _ = g.ndata.pop("agg_m")
        _ = g.ndata.pop("sum_m")
        _ = g.edata.pop("h")
        _ = g.edata.pop("w")
        _ = g.edata.pop("wh")

        return decode_node_feature

    def training_step(self, batch, batch_nb):
        """
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details
                please refer to data module class.
        """
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        gt = batch[1]
        out_graph_list = [inp_graph]
        # gt = []
        node_feat = [inp_graph.ndata["feat"][:, 2]]

        # Collect ground truth
        # for i in range(self.sequence_length):
        #     gt.append(batch[0][i].ndata['feat'][:, 2])
        # gt = torch.stack(gt)

        # Sequence rollout
        # for i in range(int((self.sequence_length-1)/(self.dim-1))):
        step_len = int((self.sequence_length - 1) / self.dim)
        for i in range(0, self.sequence_length - step_len, step_len):
            inp_graph = batch[0][i].clone()
            # inp_graph.ndata['feat'][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph

            # space osc part
            out_feat_osc = out_feat.reshape(self.col2d.size())
            f = osc2d(self.p, self.c, out_feat_osc, self.device)
            pred = f(self.base)
            node_feat.append(pred)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)):  # For each node series
            y = node_feat[1:-1, i]
            b1 = node_feat[0, i]
            b2 = node_feat[-1, i]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            pred.append(f_(self.x))
        pred = torch.permute(torch.stack(pred), (1, 0))
        loss = nn.MSELoss()(gt, pred)
        self.collection.append(loss.item())
        return loss

    def training_epoch_end(self, epoch_output):
        self.log(
            "train/loss", sum(self.collection) / len(self.collection), prog_bar=True
        )
        self.collection = []

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details
                please refer to data module class.
        """
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        gt = batch[1]
        out_graph_list = [inp_graph]
        # gt = []
        node_feat = [inp_graph.ndata["feat"][:, 2]]

        # Collect ground truth
        # for i in range(self.sequence_length):
        #     gt.append(batch[0][i].ndata['feat'][:, 2])
        # gt = torch.stack(gt)

        # Sequence rollout
        # for i in range(int((self.sequence_length-1)/(self.dim-1))):
        step_len = int((self.sequence_length - 1) / self.dim)
        for i in range(0, self.sequence_length - step_len, step_len):
            inp_graph = batch[0][i].clone()
            # inp_graph.ndata['feat'][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph

            # space osc part
            out_feat_osc = out_feat.reshape(self.col2d.size())
            f = osc2d(self.p, self.c, out_feat_osc, self.device)
            pred = f(self.base)
            node_feat.append(pred)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)):  # For each node series
            y = node_feat[1:-1, i]
            b1 = node_feat[0, i]
            b2 = node_feat[-1, i]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            pred.append(f_(self.x))
        pred = torch.permute(torch.stack(pred), (1, 0))
        loss = nn.MSELoss()(gt, pred)
        self.collection.append(loss.item())
        self.val_collection.append(loss.item())

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def validation_step_end(self, validation_step_output):
        self.log(
            "val/loss",
            sum(self.val_collection) / len(self.val_collection),
            prog_bar=False,
            batch_size=1,
        )
        self.val_collection = []

    def _get_base(self, nx, nx_sim, dx):
        x = torch.linspace(0, 1, nx_sim)
        split_x = int(nx_sim / nx)
        sx = []
        for i in range(nx + 1):
            sx.append(torch.ones(split_x) * i / nx)

        px_l0 = torch.cat(sx[:nx])
        px_l1 = torch.cat(sx[1:])
        px_l2 = torch.cat(sx[2:] + sx[-1:])

        px_r0 = torch.cat(sx[:1] + sx[: nx - 1])
        px_r1 = px_l0
        px_r2 = px_l1

        base_xl0 = (dx + 2 * (px_l1 - x)) * (x - px_l0) ** 2 / dx ** 3
        base_xr0 = (dx + 2 * (x - px_r1)) * (px_r2 - x) ** 2 / dx ** 3
        base_xl1 = (x - px_l1) * (x - px_l0) ** 2 / dx ** 2
        base_xr1 = (x - px_r1) * (px_r2 - x) ** 2 / dx ** 2

        base_start = torch.cat(
            (
                x[:split_x] * (sx[1] - x[:split_x]) ** 2 / dx ** 2,
                torch.zeros(nx_sim - split_x),
            )
        )
        base_end = torch.cat(
            (
                torch.zeros(nx_sim - split_x),
                (x[-split_x:] - sx[-1]) * (x[-split_x:] - sx[-2]) ** 2 / dx ** 2,
            )
        )
        base_list = [base_start]
        for i in range(nx - 1):
            base0 = torch.cat(
                (
                    torch.zeros(i * split_x),
                    base_xl0[i * split_x : (i + 1) * split_x],
                    base_xr0[i * split_x : (i + 1) * split_x],
                    torch.zeros((nx - 2 - i) * split_x),
                )
            )
            base1 = torch.cat(
                (
                    torch.zeros(i * split_x),
                    base_xl1[i * split_x : (i + 1) * split_x],
                    base_xr1[i * split_x : (i + 1) * split_x],
                    torch.zeros((nx - 2 - i) * split_x),
                )
            )
            base_list.append(base0)
            base_list.append(base1)
        base_list.append(base_end)
        base_x = torch.stack(base_list, dim=0)
        return base_x


if __name__ == "__main__":
    model = TimeSplineNets(sequence_length=10, dim=2)
    print(model)
