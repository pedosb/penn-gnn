import torch
from matplotlib import pyplot as plt
import numpy as np

torch.random.manual_seed(20)

from lab2_graph_filter import GraphFilter, MultiLayerGNN

filter_order = 8
learning_rate = 5e-2
n_epochs = 30
batch_size = 200
validation_ration = 0.1

graph_shift_operator, X_train, X_test, Y_train, Y_test = torch.load('lab2_dataset.pt')

train_size = int(np.floor((1 - validation_ration) * X_train.shape[0]))
X_train, X_validation = torch.split(X_train, train_size)
Y_train, Y_validation = torch.split(Y_train, train_size)


def run(task, verbose=False, save_fig=False):
    if task == 'graph_filter':
        filter_model = GraphFilter(filter_order, graph_shift_operator, 1, 1, use_activation=False)
    elif task == 'graph_perceptron':
        filter_model = GraphFilter(filter_order, graph_shift_operator, 1, 1, use_activation=True)
    elif task == 'multilayer_gnn':
        banks_order = [8, 1]
        n_filters_per_bank = [1, 1]
        filter_model = MultiLayerGNN(1, n_filters_per_bank, banks_order, graph_shift_operator)
    elif task == 'multi_feature_graph_filter':
        banks_order = [8, 1]
        n_filters_per_bank = [32, 1]
        filter_model = MultiLayerGNN(1,
                                     n_filters_per_bank,
                                     banks_order,
                                     graph_shift_operator,
                                     use_activation=False)
    elif task == 'multi_feature_two_layers_gnn':
        banks_order = [8, 1]
        n_filters_per_bank = [32, 1]
        filter_model = MultiLayerGNN(1,
                                     n_filters_per_bank,
                                     banks_order,
                                     graph_shift_operator,
                                     use_activation=True)
    elif task == 'multi_feature_three_layers_gnn':
        banks_order = [5, 5, 1]
        n_filters_per_bank = [16, 4, 1]
        filter_model = MultiLayerGNN(1,
                                     n_filters_per_bank,
                                     banks_order,
                                     graph_shift_operator,
                                     use_activation=True)

    optimizer = torch.optim.Adam(filter_model.parameters(), learning_rate)
    loss_function = torch.nn.MSELoss(reduction='mean')

    batch_loss_history = []
    validation_loss_history = []
    step = 0
    for _ in range(n_epochs):
        for batch_idx in torch.split(torch.randperm(X_train.shape[0]), batch_size):
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]

            predicted = filter_model.forward(X_batch).squeeze()
            loss = loss_function(Y_batch, predicted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_history.append((step, loss.detach().numpy()))

            step += 1
        with torch.no_grad():
            predicted = filter_model.forward(X_validation).squeeze()
            validation_loss_history.append((step, loss_function(Y_validation, predicted).numpy()))

    if save_fig or verbose:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # ax.set_yscale('log')
        ax.plot(*np.array(batch_loss_history).T, label='Batch loss')
        ax.plot(*np.array(validation_loss_history).T, label='Validation loss')
        ax.legend()

        if save_fig:
            fig.savefig(f'figures/{task}.png')

        if verbose:
            plt.show()

        plt.close()

    with torch.no_grad():
        predicted = filter_model.forward(X_test).squeeze()
        test_loss = loss_function(Y_test, predicted)
        if verbose:
            print(f'Test loss {test_loss}')
            print(f'Validation loss {validation_loss_history[-1][1]}')
            predicted = filter_model.forward(X_train)
            train_loss = loss_function(Y_train, predicted)
            print(f'Train loss {train_loss}')
            plt.show()
    return test_loss


if __name__ == '__main__':
    run('graph_perceptron', True)
