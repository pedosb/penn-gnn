import numpy as np
import torch
from matplotlib import pyplot as plt
from torchinfo import summary

torch.random.manual_seed(20)
from training import train_model, evaluate_model_loss

from lab2_graph_filter import GraphFilter, MultiLayerGNN

filter_order = 8
learning_rate = 5e-2
n_epochs = 30
batch_size = 200


def run(graph_shift_operator,
        X_train,
        X_test,
        X_validation,
        Y_train,
        Y_test,
        Y_validation,
        task,
        verbose=False,
        save=False,
        show_summary=False):
    n_nodes = X_train.shape[1]
    squeeze_feature_dims = False

    if task == 'graph_filter':
        model = GraphFilter(filter_order, graph_shift_operator, 1, 1, use_activation=False)
    elif task == 'graph_perceptron':
        model = GraphFilter(filter_order, graph_shift_operator, 1, 1, use_activation=True)
    elif task == 'multilayer_gnn':
        banks_order = [8, 1]
        n_filters_per_bank = [1, 1]
        model = MultiLayerGNN(1, n_filters_per_bank, banks_order, graph_shift_operator)
    elif task == 'multi_feature_graph_filter':
        banks_order = [8, 1]
        n_filters_per_bank = [32, 1]
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              graph_shift_operator,
                              use_activation=False)
    elif task == 'multi_feature_two_layers_gnn':
        banks_order = [8, 1]
        n_filters_per_bank = [32, 1]
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              graph_shift_operator,
                              use_activation=True)
    elif task == 'multi_feature_three_layers_gnn':
        banks_order = [5, 5, 1]
        n_filters_per_bank = [16, 4, 1]
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              graph_shift_operator,
                              use_activation=True)
    elif task == 'linear':
        squeeze_feature_dims = True
        model = torch.nn.Linear(n_nodes, n_nodes)
    elif task == 'fcnn':
        squeeze_feature_dims = True
        model = torch.nn.Sequential(torch.nn.Linear(n_nodes, 25), torch.nn.LeakyReLU(),
                                    torch.nn.Linear(25, n_nodes))

    if squeeze_feature_dims:
        X_train = torch.squeeze(X_train)
        X_test = torch.squeeze(X_test)
        X_validation = torch.squeeze(X_validation)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_function = torch.nn.MSELoss(reduction='mean')

    train_model(model, optimizer, n_epochs, batch_size, X_train, Y_train, loss_function,
                X_validation, Y_validation, task if save else None, verbose)

    if show_summary:
        summary(model)

    test_loss, _, _ = evaluate_model_loss(model, loss_function, X_test, Y_test, X_train, Y_train,
                                          X_validation, Y_validation, verbose)

    return test_loss


if __name__ == '__main__':
    run('graph_perceptron', True)
