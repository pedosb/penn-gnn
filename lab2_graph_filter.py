import math
import torch
from torch import nn

from lab2_graph_operations import filter_graph_signal


class GraphFilter(nn.Module):

    def __init__(self,
                 order,
                 graph_shift_operator,
                 n_features_in,
                 n_filters,
                 use_activation=True,
                 use_bias=False):
        """Create a optimizable graph filter

        Args:
            order (list): number of taps for the graph filter.
            graph_shift_operator (torch.tensor): GSO/adjacency matrix.
            n_features_in (int): dimension of the input to the filter.
            n_filters (int): number of filters on this layer.
            use_activation (bool, optional): use the non-linear function. Defaults to True.
        """
        super().__init__()
        self.graph_shift_operator = graph_shift_operator
        self.coefficients = nn.Parameter(torch.rand((order, n_features_in, n_filters)))
        if use_bias:
            self.bias = nn.Parameter(torch.rand((order, )))
        else:
            self.bias = torch.zeros((order, ))
        self.reset_parameters()
        if use_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(len(self.coefficients))
        self.coefficients.data.uniform_(-stdv, stdv)

    def forward(self, X):
        filtered_signal = filter_graph_signal(self.coefficients,
                                              self.graph_shift_operator,
                                              X,
                                              bias=self.bias)
        if self.activation is not None:
            return self.activation(filtered_signal)
        else:
            return filtered_signal


class MultiLayerGNN(nn.Module):

    def __init__(self,
                 n_features_in,
                 n_filters_per_bank,
                 banks_order,
                 n_readout_features_out,
                 graph_shift_operator,
                 use_activation=True,
                 use_bias=False) -> None:
        super().__init__()
        filters = []
        for n_features_out, order in zip(n_filters_per_bank, banks_order):
            filters.append(
                GraphFilter(order, graph_shift_operator, n_features_in, n_features_out,
                            use_activation, use_bias))
            n_features_in = n_features_out
        self.layers = nn.Sequential(*filters)
        self.readout = nn.Linear(n_filters_per_bank[-1], n_readout_features_out, bias=True)

    def forward(self, X):
        return self.readout(self.layers(X))

    def set_graph_shift_operator(self, new_graph_shift_operator):
        for layer in self.layers:
            if isinstance(layer, GraphFilter):
                layer.graph_shift_operator = new_graph_shift_operator
