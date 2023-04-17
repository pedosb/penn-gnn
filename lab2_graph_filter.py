import torch
from torch import nn

from lab2_graph_oprations import filter


class GraphFilter(nn.Module):

    def __init__(self, order, graph_shift_operator):
        super().__init__()
        self.graph_shift_operator = graph_shift_operator
        self.coefficients = nn.Parameter(torch.rand((order, )))

    def forward(self, X):
        filter(self.coefficients, X, self.graph_shift_operator)
