import dgl
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn.functional import interpolate
from dgl.nn.pytorch.softmax import edge_softmax
from scipy.interpolate import griddata
from dgl import batch

from src.numerics.osc import osc1d, osc2d
from src.models.components.base import BaseGNN
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class SeqGNN(BaseGNN):
    ''' GNN with Sequence input '''

    default_input_feats = ['feat', 'x', 'y', 'time', 'mask']

    def __init__(self, sequence_length, input_feats=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feats = input_feats if input_feats else self.default_input_feats
        self.sequence_length = sequence_length
        self.collection = []
        self.val_collection = []

    def forward(self, g):
        # Encoder
        encoder_data = torch.cat([g.ndata[f] for f in self.in_feats], dim=-1)
        enc_n_feat = self.encoder(encoder_data)

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata['dist'])

        # Message Passing
        g.ndata['h'] = softmax_n_feat
        g.edata['h'] = g.edata['dist'] # NOTE: was feat

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['dist'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        if 'mask' in self.in_feats:
            decode_node_feature = decode_node_feature * (1-g.ndata['mask'][:, None]) # 1 = invalid, we want to multiply by 0

        # (Important) Delete temp features
        _ = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        _ = g.ndata.pop('sum_m')
        _ = g.edata.pop('h')
        _ = g.edata.pop('w')
        _ = g.edata.pop('wh')

        return decode_node_feature

    def training_step(self, batch, batch_nb): 
        '''
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details 
                please refer to data module class.
        '''
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        out_graph_list = [inp_graph]
        mask = (1 - inp_graph.ndata['mask'][None]) if 'mask' in self.in_feats else None

        gt = []
        node_feat = [inp_graph.ndata['feat']]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'])
        gt = torch.stack(gt)

        # Sequence rollout
        for i in range(self.sequence_length-1):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))[:,-1,:]
            del inp_graph
            node_feat.append(out_feat)

        # Apply OSC
        pred = torch.stack(node_feat)

        # Mask out the invalid region
        if mask is not None:
            loss = nn.MSELoss()(gt * mask, pred * mask)
        else:
            loss = nn.MSELoss()(gt, pred)
        self.collection.append(loss.item())
        return loss


    def training_epoch_end(self, epoch_output):
        self.log('train/loss', sum(self.collection)/len(self.collection), prog_bar=True)
        self.collection = []

    def validation_step(self, batch, batch_idx):
        '''
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details 
                please refer to data module class.
        '''
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        gt = []
        mask = (1 - inp_graph.ndata['mask'][None]) if 'mask' in self.in_feats else None
        node_feat = [inp_graph.ndata['feat']]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'])
        gt = torch.stack(gt)

        # Sequence rollout
        for i in range(self.sequence_length-1):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))[:, -1, :] # only last state in time
            del inp_graph
            node_feat.append(out_feat)

        pred = torch.stack(node_feat)

        # Mask out the invalid region
        if mask is not None:
            loss = nn.MSELoss()(gt * mask, pred * mask)
        else:
            loss = nn.MSELoss()(gt, pred)
        self.val_collection.append(loss.item())

    def validation_step_end(self, validation_step_output):
        self.log('val/loss', sum(self.val_collection)/len(self.val_collection), prog_bar=False, batch_size=1)
        self.val_collection = []



class TimeSplineNets(BaseGNN):
    ''' SplineGraphNets with only time oriented collocation'''

    default_input_feats = ['feat', 'x', 'y', 'time', 'mask']

    def __init__(self, sequence_length, dim, input_feats=None ,**kwargs):
        super().__init__(**kwargs)
        
        self.in_feats = input_feats if input_feats else self.default_input_feats
        self.collection = []
        self.val_collection = []

        # Setup OSC1d
        self.sequence_length = sequence_length
        self.x = torch.tensor(np.linspace(0, 1, sequence_length, endpoint=True), dtype=torch.float32)
        self.dim = dim
        self.p = torch.tensor([0, 1])
        self.c = torch.tensor(np.linspace(0, 1, dim+1, endpoint=True)[1:-1], dtype=torch.float32)

    def forward(self, g):
        # Encoder
        encoder_data = torch.cat([g.ndata[f] for f in self.in_feats], dim=-1)
        enc_n_feat = self.encoder(encoder_data)

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata['dist'])

        # Message Passing
        g.ndata['h'] = softmax_n_feat
        g.edata['h'] = g.edata['dist'] # NOTE: was feat

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['dist'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        if 'mask' in self.in_feats:
            decode_node_feature = decode_node_feature * (1-g.ndata['mask'][:, None]) # 1 = invalid, we want to multiply by 0

        # (Important) Delete temp features
        _ = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        _ = g.ndata.pop('sum_m')
        _ = g.edata.pop('h')
        _ = g.edata.pop('w')
        _ = g.edata.pop('wh')

        return decode_node_feature

    def training_step(self, batch, batch_nb): 
        '''
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details 
                please refer to data module class.
        '''
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        out_graph_list = [inp_graph]
        mask = (1 - inp_graph.ndata['mask'][None]) if 'mask' in self.in_feats else None

        gt = []
        node_feat = [inp_graph.ndata['feat']]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'])
        gt = torch.stack(gt)

        # Sequence rollout
        # for i in range(int((self.sequence_length-1)/(self.dim-1))):
        step_len = int((self.sequence_length-1)/self.dim)
        for i in range(0, self.sequence_length-step_len, step_len):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))[:,-1,:]
            del inp_graph
            node_feat.append(out_feat)

        # Apply OSC
        pred = self._time_osc(node_feat)

        # Mask out the invalid region
        if mask is not None:
            loss = nn.MSELoss()(gt * mask, pred * mask)
        else:
            loss = nn.MSELoss()(gt, pred)
        self.collection.append(loss.item())
        return loss

    def training_epoch_end(self, epoch_output):
        self.log('train/loss', sum(self.collection)/len(self.collection), prog_bar=True)
        self.collection = []

    def validation_step(self, batch, batch_idx):
        '''
        Args:
            batch <list of list> [batch_size, sequence_length]: mostly the batch_size
                will be set to 1, so the input is a sequence of graph. More details 
                please refer to data module class.
        '''
        loss = torch.tensor(0, dtype=torch.float32)
        inp_graph = batch[0][0]
        gt = []
        mask = (1 - inp_graph.ndata['mask'][None]) if 'mask' in self.in_feats else None
        node_feat = [inp_graph.ndata['feat']]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'])
        gt = torch.stack(gt)

        # Sequence rollout
        step_len = int((self.sequence_length-1)/self.dim)
        for i in range(0, self.sequence_length-step_len, step_len):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))[:, -1, :] # only last state in time
            del inp_graph
            node_feat.append(out_feat)

        pred = self._time_osc(node_feat)

        # Mask out the invalid region
        if mask is not None:
            loss = nn.MSELoss()(gt * mask, pred * mask)
        else:
            loss = nn.MSELoss()(gt, pred)
        self.val_collection.append(loss.item())

    def _time_osc(self, node_feats):
        """
        T x N x D
        Apply time-oriented OSC to each node via list comprehension
        """
        node_feats = torch.stack(node_feats)

        def _apply_osc(feat):
            y = feat[1:-1]
            b1 = feat[0]
            b2 = feat[-1]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            return f_(self.x)

        def _apply_osc_at_dim(node_feats_at_dim):
            pred = [_apply_osc(node_feats_at_dim[:, j]) for j in range(node_feats_at_dim.shape[-1])]
            pred = torch.permute(torch.stack(pred), (1, 0))
            return pred

        pred = [_apply_osc_at_dim(node_feats[:, :, i]) for i in range(node_feats.shape[-1])]
        pred = torch.stack(pred, dim=-1)
        return pred

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def validation_step_end(self, validation_step_output):
        self.log('val/loss', sum(self.val_collection)/len(self.val_collection), prog_bar=False, batch_size=1)
        self.val_collection = []


if __name__ == '__main__':
    model = TimeSplineNets(sequence_length=10, dim=2)
    print(model)