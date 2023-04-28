import numpy as np
import torch
from matplotlib import pyplot as plt
from torchinfo import summary

torch.random.manual_seed(20)

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

    batch_loss_history = []
    validation_loss_history = []
    step = 0
    for _ in range(n_epochs):
        for batch_idx in torch.split(torch.randperm(X_train.shape[0]), batch_size):
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]

            predicted = model.forward(X_batch).squeeze()
            loss = loss_function(Y_batch, predicted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_history.append((step, loss.detach().numpy()))

            step += 1
        with torch.no_grad():
            predicted = model.forward(X_validation).squeeze()
            validation_loss_history.append((step, loss_function(Y_validation, predicted).numpy()))

    if save or verbose:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # ax.set_yscale('log')
        ax.plot(*np.array(batch_loss_history).T, label='Batch loss')
        ax.plot(*np.array(validation_loss_history).T, label='Validation loss')
        ax.legend()

        if save:
            fig.savefig(f'figures/{task}.png')

        if verbose:
            plt.show()

        plt.close()

    if save:
        torch.save(model, f'models/{task}.pt')

    with torch.no_grad():
        predicted = model.forward(X_test).squeeze()
        test_loss = loss_function(Y_test, predicted)
        if verbose:
            print(f'Test loss {test_loss}')
            print(f'Validation loss {validation_loss_history[-1][1]}')
            predicted = model.forward(X_train)
            train_loss = loss_function(Y_train, predicted)
            print(f'Train loss {train_loss}')
            plt.show()

    if show_summary:
        summary(model)

    return test_loss


if __name__ == '__main__':
    run('graph_perceptron', True)
