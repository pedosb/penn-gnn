# %%
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from lab2_graph_filter import GraphFilter, MultiLayerGNN
from lab3_dataset import (create_samples_from_ratings, load_movie_dataset,
                          remove_users_not_rating_movie, split_train_test_users)
from lab3_similarity_graph import (compute_movie_similarity_graph, stratify_and_normalize)
from training import evaluate_model_loss, train_model
from utils import random

torch.random.manual_seed(20)


# %%
def make_estimate_single_movie_loss(movie_index):
    mse = torch.nn.MSELoss()
    return lambda *y: mse(*y) / 2


def run_experiment(adjacency_matrix,
                   target_movie,
                   X_train,
                   X_test,
                   Y_train,
                   Y_test,
                   task,
                   save=False,
                   verbose=False):
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
                save_prefix=task if save else None,
                verbose=verbose)

    evaluate_model_loss(model, loss_function, X_test, Y_test, X_train, Y_train, verbose=verbose)


for i in range(1):
    contact_index, adjacency_matrix, X_train, X_test, Y_train, Y_test = torch.load(
        f'datasets/lab3_similarity_{i:02d}.pt')
    run_experiment(adjacency_matrix,
                   contact_index,
                   X_train,
                   X_test,
                   Y_train,
                   Y_test,
                   task='graph_filter',
                   save=False,
                   verbose=True)
