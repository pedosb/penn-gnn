# %%
import torch
from torchinfo import summary

from lab2_graph_filter import MultiLayerGNN
from training import evaluate_model_loss, train_model
from utils import random

torch.random.manual_seed(20)


# %%
def make_estimate_single_movie_mse(movie_index):
    mse = torch.nn.MSELoss()
    return lambda y, pred: mse(y[..., movie_index], pred[..., movie_index])


def run_experiment(target_movie,
                   adjacency_matrix,
                   X_train,
                   X_test,
                   X_validation,
                   Y_train,
                   Y_test,
                   Y_validation,
                   task,
                   save_prefix=None,
                   verbose=False,
                   show_summary=False):
    n_nodes = X_train.shape[1]
    squeeze_feature_dims = False
    learning_rate = 5e-2
    n_epochs = 40
    batch_size = 5

    if task == 'graph_filter':
        banks_order = [5, 1]
        n_filters_per_bank = [64, 1]
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              adjacency_matrix,
                              use_activation=False,
                              use_bias=True)
    elif task == 'linear':
        squeeze_feature_dims = True
        model = torch.nn.Linear(n_nodes, n_nodes)
    elif task == 'fcnn':
        squeeze_feature_dims = True
        model = torch.nn.Sequential(torch.nn.Linear(n_nodes, 64), torch.nn.LeakyReLU(),
                                    torch.nn.Linear(64, n_nodes))
    elif task == 'gnn':
        banks_order = [5]
        n_filters_per_bank = [64]
        n_features_out = 1
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              n_features_out,
                              adjacency_matrix,
                              use_activation=True,
                              use_bias=True)
    elif task == '2l_gnn':
        banks_order = [5, 5]
        n_filters_per_bank = [64, 32]
        n_features_out = 1
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              n_features_out,
                              adjacency_matrix,
                              use_activation=True,
                              use_bias=True)

    if squeeze_feature_dims:
        X_train = torch.squeeze(X_train)
        X_validation = torch.squeeze(X_validation)
        X_test = torch.squeeze(X_test)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_model(model,
                optimizer,
                n_epochs,
                batch_size,
                X_train,
                Y_train,
                lambda *y: make_estimate_single_movie_mse(target_movie)(*y) / 2,
                X_validation,
                Y_validation,
                save_prefix=save_prefix,
                verbose=verbose)

    test_loss, train_loss, _ = evaluate_model_loss(
        model,
        lambda *y: torch.sqrt(make_estimate_single_movie_mse(target_movie)(*y)),
        X_test,
        Y_test,
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        verbose=verbose)

    if show_summary:
        summary(model)

    return test_loss, train_loss
