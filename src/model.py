from json import load
import sys; sys.path.append('.')
import copy
import dgl
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from dgl.nn.pytorch.softmax import edge_softmax
from src.osc import osc1d


class BaseGNN(LightningModule):
    ''' Base GNN Model (GEN) '''
    def __init__(self, lr=1e-3):
        super(BaseGNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.collection = []
        self.lr = lr
    
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
        self.log('train_loss', sum(self.collection)/len(self.collection), prog_bar=True)
        self.collection = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def softmax(self, g, node_feature, edge_feature):
        g.ndata['h'] = node_feature
        g.edata['w'] = edge_softmax(g, edge_feature)
        g.apply_edges(func=dgl.function.e_mul_u('w', 'h', 'wh'))
        g.update_all(message_func=dgl.function.copy_e('wh', 'm'), reduce_func=dgl.function.sum('m', 'sum_m'))
        softmax_node_feature = g.ndata['sum_m']
        return softmax_node_feature

    def node_update_func(self, nodes):
        ''' Update Node Feature '''
        aggregation_message = nodes.data['agg_m']
        node_feature = nodes.data['h']
        node_model_input = torch.cat([aggregation_message, node_feature], dim=-1)
        updated_node_feature = self.node_update(node_model_input)
        return {'h': updated_node_feature}

    def edge_update_func(self, edges):
        ''' Update Edge Feature '''
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}


class SeqGNN(LightningModule):
    ''' GNN with Sequence input '''
    def __init__(self, sequence_length, lr=1e-3):
        super(SeqGNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.collection = []
        self.val_collection = []
        self.sequence_length = sequence_length
        self.lr = lr

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
        self.log('train_loss', sum(self.collection)/len(self.collection), prog_bar=True)
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
        self.log('val_loss', sum(self.val_collection)/len(self.val_collection), prog_bar=False, batch_size=1)
        self.val_collection = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def softmax(self, g, node_feature, edge_feature):
        g.ndata['h'] = node_feature
        g.edata['w'] = edge_softmax(g, edge_feature)
        g.apply_edges(func=dgl.function.e_mul_u('w', 'h', 'wh'))
        g.update_all(message_func=dgl.function.copy_e('wh', 'm'), reduce_func=dgl.function.sum('m', 'sum_m'))
        softmax_node_feature = g.ndata['sum_m']
        return softmax_node_feature

    def node_update_func(self, nodes):
        ''' Update Node Feature '''
        aggregation_message = nodes.data['agg_m']
        node_feature = nodes.data['h']
        node_model_input = torch.cat([aggregation_message, node_feature], dim=-1)
        updated_node_feature = self.node_update(node_model_input)
        return {'h': updated_node_feature}

    def edge_update_func(self, edges):
        ''' Update Edge Feature '''
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}


class TimeSplineNets(LightningModule):
    ''' SplineGraphNets with time oriented collocation'''
    def __init__(self, sequence_length, dim, lr=1e-3):
        super(TimeSplineNets, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        )
        self.collection = []
        self.val_collection = []
        self.lr = lr

        # Setup OSC1d
        self.sequence_length = sequence_length
        self.x = torch.tensor(np.linspace(0, 1, sequence_length, endpoint=True), dtype=torch.float32)
        self.dim = dim
        self.p = torch.tensor([0, 1])
        self.c = torch.tensor(np.linspace(0, 1, dim+1, endpoint=True)[1:-1], dtype=torch.float32)

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
        out_graph_list = [inp_graph]
        gt = []
        node_feat = [inp_graph.ndata['feat'][:, 2]]

        # Collect ground truth
        for i in range(self.sequence_length):
            gt.append(batch[0][i].ndata['feat'][:, 2])
        gt = torch.stack(gt)

        # Sequence rollout
        step_len = int((self.sequence_length-1)/self.dim)
        for i in range(0, self.sequence_length-step_len, step_len):
            inp_graph = batch[0][i].clone()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)): # For each node series
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
        self.log('train_loss', sum(self.collection)/len(self.collection), prog_bar=True)
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
        step_len = int((self.sequence_length-1)/self.dim)
        for i in range(0, self.sequence_length-step_len, step_len):
            inp_graph = batch[0][i].clone()
            inp_graph.ndata['feat'][:, 2] = node_feat[-1].clone().detach()
            out_feat = torch.squeeze(self(inp_graph))
            del inp_graph
            node_feat.append(out_feat)

        # Time-oriented OSC for each node
        node_feat = torch.stack(node_feat)
        pred = []
        for i in range(node_feat.size(1)): # For each node series
            y = node_feat[1:-1, i]
            b1 = node_feat[0, i]
            b2 = node_feat[-1, i]
            f_ = osc1d(self.p, self.c, y, b1, b2, self.device)
            pred.append(f_(self.x))
        pred = torch.permute(torch.stack(pred), (1, 0))
        loss = nn.MSELoss()(gt, pred)
        self.val_collection.append(loss.item())

    def validation_step_end(self, validation_step_output):
        self.log('val_loss', sum(self.val_collection)/len(self.val_collection), prog_bar=False, batch_size=1)
        self.val_collection = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def softmax(self, g, node_feature, edge_feature):
        g.ndata['h'] = node_feature
        g.edata['w'] = edge_softmax(g, edge_feature)
        g.apply_edges(func=dgl.function.e_mul_u('w', 'h', 'wh'))
        g.update_all(message_func=dgl.function.copy_e('wh', 'm'), reduce_func=dgl.function.sum('m', 'sum_m'))
        softmax_node_feature = g.ndata['sum_m']
        return softmax_node_feature

    def node_update_func(self, nodes):
        ''' Update Node Feature '''
        aggregation_message = nodes.data['agg_m']
        node_feature = nodes.data['h']
        node_model_input = torch.cat([aggregation_message, node_feature], dim=-1)
        updated_node_feature = self.node_update(node_model_input)
        return {'h': updated_node_feature}

    def edge_update_func(self, edges):
        ''' Update Edge Feature '''
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}
