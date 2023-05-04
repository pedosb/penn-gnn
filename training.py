from matplotlib import pyplot as plt
import numpy as np
import torch


def train_model(
    model,
    optimizer,
    n_epochs,
    batch_size,
    X_train,
    Y_train,
    loss_function,
    X_validation=None,
    Y_validation=None,
    save_prefix=None,
    verbose=False,
):
    save = save_prefix is not None
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
        if X_validation is not None:
            with torch.no_grad():
                predicted = model.forward(X_validation).squeeze()
                validation_loss_history.append((step, loss_function(Y_validation,
                                                                    predicted).numpy()))

    if save or verbose:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(*np.array(batch_loss_history).T, label='Batch loss')
        if X_validation is not None:
            ax.plot(*np.array(validation_loss_history).T, label='Validation loss')
        ax.legend()

        if save:
            fig.savefig(f'figures/{save_prefix}.png')

        elif verbose:
            plt.show()

        plt.close()

    if save:
        torch.save(model, f'models/{save_prefix}.pt')


def evaluate_model_loss(model,
                        loss_function,
                        X_test,
                        Y_test,
                        X_train=None,
                        Y_train=None,
                        X_validation=None,
                        Y_validation=None,
                        verbose=False):
    validation_loss = torch.nan
    train_loss = torch.nan
    with torch.no_grad():
        predicted = model.forward(X_test).squeeze()
        test_loss = loss_function(Y_test, predicted)
        if X_validation is not None:
            predicted = model.forward(X_validation).squeeze()
            validation_loss = loss_function(Y_validation, predicted)
        if X_train is not None:
            predicted = model.forward(X_train).squeeze()
            train_loss = loss_function(Y_train, predicted)

        if verbose:
            print(f'Test loss {test_loss}')
            if X_train is not None:
                print(f'Train loss {train_loss}')
            if X_validation is not None:
                print(f'Validation loss {validation_loss}')

    return test_loss, train_loss, validation_loss
