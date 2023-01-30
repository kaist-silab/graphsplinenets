from typing import List

import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP to be used inside GNN modules
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_prob: float = 0.0,
        num_neurons: List[int] = [128, 128],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "none",
        output_norm: str = "none",
        bias: bool = True,
    ):

        super().__init__()
        assert input_norm in ["batch", "layer", "none"]
        assert output_norm in ["batch", "layer", "none"]

        layers = []
        if input_norm != "none":
            layers.append(self._get_norm_layer(input_norm, input_dim))

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

            last_layer = i != len(input_dims) - 1
            if last_layer:
                act_str = hidden_act
            else:
                act_str = out_act

            act = getattr(nn, act_str)()

            layers.append(act)
            if dropout_prob > 0.0 and not last_layer:
                layers.append(nn.Dropout(dropout_prob))

        if output_norm != "none":
            layers.append(self._get_norm_layer(output_norm, input_dim))
        self.net = nn.Sequential(*tuple(layers))

    def forward(self, xs):
        return self.net(xs)

    def _get_norm_layer(self, norm_method, dim):
        if norm_method == "batch":
            norm = nn.BatchNorm1d(dim)
        elif norm_method == "layer":
            norm = nn.LayerNorm(dim)
        else:
            raise RuntimeError("Normalization method not supported")
        return norm
