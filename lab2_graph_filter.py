import math
import torch
from torch import nn

from lab2_graph_operations import filter


class GraphFilter(nn.Module):

    def __init__(self, order, graph_shift_operator, use_activation=True):
        """Create a optimizable graph filter

        Args:
            order (list): number of taps for the graph filter.
            graph_shift_operator (torch.tensor): GSO/adjacency matrix.
            use_activation (bool, optional): use the non-linear function. Defaults to True.
        """
        super().__init__()
        self.graph_shift_operator = graph_shift_operator
        self.coefficients = nn.Parameter(torch.rand((order, )))
        self.reset_parameters()
        if use_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(len(self.coefficients))
        self.coefficients.data.uniform_(-stdv, stdv)

    def forward(self, X):
        filtered_signal = filter(self.coefficients, X, self.graph_shift_operator)
        if self.activation is not None:
            return self.activation(filtered_signal)
        else:
            return filtered_signal


class MultiLayerGNN(nn.Module):

    def __init__(self, filters_order, graph_shift_operator) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[GraphFilter(order, graph_shift_operator) for order in filters_order])

    def forward(self, X):
        return self.layers(X)
