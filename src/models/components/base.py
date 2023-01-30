import dgl
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from dgl.nn.pytorch.softmax import edge_softmax
import hydra 

from src.utils import pylogger
from src.optim.param_grouping import group_parameters_for_optimizer


log = pylogger.get_pylogger(__name__)


class BaseGNN(LightningModule):
    """
    Template GNN model with Hydra
    """
    def __init__(self, cfg, model_cfg=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model
        self.instantiate_model()

    def instantiate_model(self):
        log.info(f"Instantiating encoder <{self.model_cfg.encoder._target_}>")
        self.encoder = hydra.utils.instantiate(self.model_cfg.encoder, _recursive_=False)
        log.info(f"Instantiating encoder <{self.model_cfg.decoder._target_}>")
        self.decoder = hydra.utils.instantiate(self.model_cfg.decoder, _recursive_=False)
        log.info(f"Instantiating encoder <{self.model_cfg.node_update._target_}>")
        self.node_update = hydra.utils.instantiate(self.model_cfg.node_update, _recursive_=False)
        log.info(f"Instantiating encoder <{self.model_cfg.edge_update._target_}>")
        self.edge_update = hydra.utils.instantiate(self.model_cfg.edge_update, _recursive_=False)

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            parameters = self.parameters() # this will train task specific parameters such as Retrieval head for AAN

        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            log.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}

    def softmax(self, g, node_feature, edge_feature):
        g.ndata['h'] = node_feature
        g.edata['w'] = edge_softmax(g, edge_feature)
        g.apply_edges(func=dgl.function.e_mul_u('w', 'h', 'wh'))
        g.update_all(message_func=dgl.function.copy_e('wh', 'm'), reduce_func=dgl.function.sum('m', 'sum_m'))
        softmax_node_feature = g.ndata['sum_m']
        return softmax_node_feature

    def edge_update_func(self, edges):
        ''' Update Edge Feature '''
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}

    def node_update_func(self, nodes):
        ''' Update Node Feature '''
        aggregation_message = nodes.data['agg_m']
        node_feature = nodes.data['h']
        node_model_input = torch.cat([aggregation_message, node_feature], dim=-1)
        updated_node_feature = self.node_update(node_model_input)
        return {'h': updated_node_feature}

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implement in child class")

    def training_step(self, *args, **kwargs): 
        raise NotImplementedError("Implement in child class")

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError("Implement in child class")
    
    def test_step(self, *args, **kwargs):
        raise NotImplementedError("Implement in child class")
    