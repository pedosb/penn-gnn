# %%
import torch
from matplotlib import pyplot as plt
import numpy as np

from lab2_graph_filter import GraphFilter
from lab2_generation import generate_dataset

# %%
filter_order = 8
learning_rate = 5e-2
n_epochs = 30
batch_size = 200
validation_ration = 0.1

graph_shift_operator, X_train, X_test, Y_train, Y_test = [
    torch.tensor(v) for v in generate_dataset()
]
train_size = int(np.floor((1 - validation_ration) * X_train.shape[0]))
X_train, X_validation = torch.split(X_train, train_size)
Y_train, Y_validation = torch.split(Y_train, train_size)

filter_model = GraphFilter(filter_order, graph_shift_operator)

optimizer = torch.optim.Adam(filter_model.parameters(), learning_rate)
loss_function = torch.nn.MSELoss(reduction='mean')

# %%
batch_loss_history = []
validation_loss_history = []
step = 0
for epoch in range(n_epochs):
    for batch_idx in torch.split(torch.randperm(X_train.shape[0]), batch_size):
        X_batch = X_train[batch_idx]
        Y_batch = Y_train[batch_idx]
        predicted = filter_model.forward(X_batch)
        loss = loss_function(Y_batch, predicted)
        batch_loss_history.append(loss.detach().numpy())
        filter_model.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    with torch.no_grad():
        predicted = filter_model.forward(X_validation)
        validation_loss_history.append((step, loss_function(Y_validation, predicted).numpy()))

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# ax.set_yscale('log')
ax.plot(batch_loss_history, label='Batch loss')
ax.plot(*np.array(validation_loss_history).T, label='Validation loss')
ax.legend()

with torch.no_grad():
    predicted = filter_model.forward(X_test)
    test_loss = loss_function(Y_test, predicted)
    print(f'Test loss {test_loss}')
    print(f'Validation loss {validation_loss_history[-1][1]}')
    predicted = filter_model.forward(X_train)
    train_loss = loss_function(Y_train, predicted)
    print(f'Train loss {train_loss}')
