import dgl
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from dgl.nn.pytorch.softmax import edge_softmax

from src.models.components.base import BaseGNN


class GEN(BaseGNN):
    ''' Simple GNN Model (GEN) '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collection = []

    def forward(self, g):
        # Encoder
        enc_n_feat = self.encoder(g.ndata['feat'])

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata['dist'])

        # Message Passing
        g.ndata['h'] = softmax_n_feat
        g.edata['h'] = g.edata['feat'] 

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['dist'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # (Important) Delete temp features
        _ = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        _ = g.ndata.pop('sum_m')
        _ = g.edata.pop('h')
        _ = g.edata.pop('w')
        _ = g.edata.pop('wh')

        return decode_node_feature

    def training_step(self, batch, batch_nb): 
        inp_graph = batch[0]
        tar_feat = batch[1].ndata['feat'][:, 2]
        out_feat = torch.squeeze(self(inp_graph))
        loss = nn.MSELoss()(out_feat, tar_feat)
        self.collection.append(loss.item())
        return loss

    def training_epoch_end(self, epoch_output):
        self.log('train/loss', sum(self.collection)/len(self.collection), prog_bar=True)
        self.collection = []


class SeqGNN(BaseGNN):
    ''' GNN with Sequence input '''
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.collection = []
        self.val_collection = []

    def forward(self, g):
        # Encoder
        enc_n_feat = self.encoder(g.ndata['feat'])

        # Softmax Encoder Feature
        softmax_n_feat = self.softmax(g, enc_n_feat, g.edata['dist'])

        # Message Passing
        g.ndata['h'] = softmax_n_feat
        g.edata['h'] = g.edata['feat'] 

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Message Passing Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['dist'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

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
        gt = []
        node_feat = [inp_graph.ndata['feat'][:, 2]]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'][:, 2])
        gt = torch.stack(gt)

        # Sequence rollout
        for i in range(self.sequence_length-1):
            inp_graph = batch[0][i].clone()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        loss = nn.MSELoss()(gt, node_feat)
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
        node_feat = [inp_graph.ndata['feat'][:, 2]]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'][:, 2])
        gt = torch.stack(gt)

        # Sequence rollout
        for i in range(self.sequence_length-1):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        loss = nn.MSELoss()(gt, node_feat)
        self.val_collection.append(loss.item())

    def validation_step_end(self, validation_step_output):
        self.log('val/loss', sum(self.val_collection)/len(self.val_collection), prog_bar=False, batch_size=1)
        self.val_collection = []