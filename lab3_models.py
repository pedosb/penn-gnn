# %%
import torch
from torchinfo import summary

from lab2_graph_filter import MultiLayerGNN
from training import evaluate_model_loss, train_model
from utils import random

torch.random.manual_seed(20)


# %%
def make_estimate_single_movie_loss(movie_index):
    mse = torch.nn.MSELoss()
    return lambda *y: mse(*y) / 2


def run_experiment(target_movie,
                   adjacency_matrix,
                   X_train,
                   X_test,
                   Y_train,
                   Y_test,
                   task,
                   save_prefix=None,
                   verbose=False,
                   show_summary=False):
    n_nodes = X_train.shape[1]
    squeeze_feature_dims = False
    learning_rate = 5e-2
    n_epochs = 80
    batch_size = 200

    if task == 'graph_filter':
        banks_order = [5, 1]
        n_filters_per_bank = [64, 1]
        model = MultiLayerGNN(1,
                              n_filters_per_bank,
                              banks_order,
                              adjacency_matrix,
                              use_activation=False)
    elif task == 'linear':
        squeeze_feature_dims = True
        model = torch.nn.Linear(n_nodes, n_nodes)

    if squeeze_feature_dims:
        X_train = torch.squeeze(X_train)
        X_test = torch.squeeze(X_test)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_function = make_estimate_single_movie_loss(target_movie)

    train_model(model,
                optimizer,
                n_epochs,
                batch_size,
                X_train,
                Y_train,
                loss_function,
                save_prefix=save_prefix,
                verbose=verbose)

    test_loss, _, _ = evaluate_model_loss(model,
                                          torch.nn.MSELoss(),
                                          X_test,
                                          Y_test,
                                          X_train,
                                          Y_train,
                                          verbose=verbose)

    if show_summary:
        summary(model)

    return test_loss
