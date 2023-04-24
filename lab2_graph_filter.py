import math
import torch
from torch import nn

from lab2_graph_operations import filter


class GraphFilter(nn.Module):

    def __init__(self, order, graph_shift_operator, n_features_in, n_filters, use_activation=True):
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
        self.reset_parameters()
        if use_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(len(self.coefficients))
        self.coefficients.data.uniform_(-stdv, stdv)

    def forward(self, X):
        filtered_signal = filter(self.coefficients, self.graph_shift_operator, X)
        if self.activation is not None:
            return self.activation(filtered_signal)
        else:
            return filtered_signal


class MultiLayerGNN(nn.Module):

    def __init__(self,
                 n_features_in,
                 n_filters_per_bank,
                 banks_order,
                 graph_shift_operator,
                 use_activation=True) -> None:
        super().__init__()
        filters = []
        for n_features_out, order in zip(n_filters_per_bank, banks_order):
            filters.append(
                GraphFilter(order, graph_shift_operator, n_features_in, n_features_out,
                            use_activation))
            n_features_in = n_features_out
        self.layers = nn.Sequential(*filters)

    def forward(self, X):
        return self.layers(X)
