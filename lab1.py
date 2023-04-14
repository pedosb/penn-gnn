# %%
import numpy as np
from matplotlib import pyplot as plt
import torch

HAVE_TIME_TO_RUN_LONGER = False

# %%
m = 20
n = 30


def generate_coefficients_matrix(m, n):
    return np.random.binomial(1, 1 / m, (m, n))


def generate_half_energy_vector(length):
    return np.random.normal(0, np.sqrt(.5 / length), length)


def generate_linear_set(A, q):
    m, n = A.shape
    X = []
    Y = []
    for _ in range(q):
        X.append(generate_half_energy_vector(n))
        w = generate_half_energy_vector(m)
        Y.append(A @ X[-1] + w)

    return np.array(X), np.array(Y)


def generate_sign_set(A, q):
    """Generate a set based on sign(Ax + w)

    The generated y does not have unitary energy anymore. For that, we would have to introduce a scaling factor on the
    sign, because it is always 1 or -1.
    """
    X, Y = generate_linear_set(A, q)
    return X, np.sign(Y)


x, y = generate_sign_set(generate_coefficients_matrix(m, n), 100)
# %%
np.mean(np.sum(y**2, axis=1))


# %%
def solve_closed_erm(m, n, Q, dataset_func):
    """
    X and Y here have samples in each row (Q x n)
    """

    A = generate_coefficients_matrix(m, n)
    X, Y = dataset_func(A, Q)
    X_test, Y_test = dataset_func(A, Q)

    H = Y.T @ X @ np.linalg.inv(X.T @ X)

    Y_hat_train = (H @ X.T).T
    Y_hat_test = (H @ X_test.T).T
    train_loss = np.mean(np.sum((Y - Y_hat_train)**2, axis=1))
    test_loss = np.mean(np.sum((Y_test - Y_hat_test)**2, axis=1))
    print(f'Train {train_loss}')
    print(f'Test {test_loss}')
    y_hat_energy = np.mean(np.sum((Y_hat_test)**2, axis=1))
    print(f'Average y hat test energy {y_hat_energy}')


solve_closed_erm(m=10**2, n=10**2, Q=10**3, dataset_func=generate_linear_set)
solve_closed_erm(m=10**2, n=10**2, Q=10**3, dataset_func=generate_sign_set)
if HAVE_TIME_TO_RUN_LONGER:
    solve_closed_erm(m=10**4, n=10**4, Q=10**3, dataset_func=generate_linear_set)


# %%
def gradient_linear_model_norm_loss(H, X, Y):
    """The average gradient of 1/2 ||y-Hx||_2^2
    T is used because capital X is x but with samples in the rows instead of columns.
    """
    return ((H @ X.T - Y.T) @ X) / X.shape[0]


def solve_linear_sgd():
    n_output = 10**2
    n_input = 10**2
    n_samples = 10**3
    n_steps = 200
    learning_rate = 6
    batch_size = 32
    A = generate_coefficients_matrix(n_output, n_input)
    X, Y = generate_linear_set(A, n_samples)
    X_test, Y_test = generate_linear_set(A, n_samples)

    H = np.random.rand(*A.shape)

    def loss(Y_real, Y_predicted):
        """Lines have samples
        """
        return np.mean(np.sum((Y_real - Y_predicted)**2, axis=1))

    loss_train = []
    loss_test = []

    for step in range(n_steps):
        batch_idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_idx, :]
        Y_batch = Y[batch_idx, :]

        loss_train.append(loss(Y, (H @ X.T).T))
        loss_test.append(loss(Y_test, (H @ X_test.T).T))

        H = H - learning_rate * gradient_linear_model_norm_loss(H, X_batch, Y_batch)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_train, label='Train')
    ax.plot(loss_test, label='Test')
    ax.legend()

    print(f'Train {loss_train[-1]}')
    print(f'Test {loss_test[-1]}')


solve_linear_sgd()


# %%
class TorchLinearModel():

    def __init__(self, n_input, n_output):
        self.H = torch.rand(n_output, n_input, requires_grad=True)  # type: torch.Tensor

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            return self.forward(X)

    def forward(self, X: torch.Tensor):
        return torch.matmul(self.H, X.T).T

    def loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(X)
        loss = torch.mean(torch.sum((Y - prediction)**2, dim=1))
        return loss

    def backward(self, X: torch.Tensor, Y: torch.Tensor):
        loss = self.loss(X, Y)
        loss.backward()
        return loss

    def step(self, X: torch.Tensor, Y: torch.Tensor, learning_rate=1) -> torch.Tensor:
        if self.H.grad is not None:
            self.H.grad.zero_()
        loss = self.backward(X, Y)
        with torch.no_grad():
            self.H.add_(self.H.grad, alpha=-learning_rate)
        return loss


def solve_linear_sgd_torch():
    n_output = 10**2
    n_input = 10**2
    n_samples = 10**3
    n_steps = 200
    learning_rate = 6
    batch_size = 32
    A = generate_coefficients_matrix(n_output, n_input)
    X, Y = generate_linear_set(A, n_samples)
    X_test, Y_test = generate_linear_set(A, n_samples)

    X, Y, X_test, Y_test = torch.tensor([X, Y, X_test, Y_test], dtype=torch.float)

    model = TorchLinearModel(n_input, n_output)

    loss_train = []
    loss_test = []

    for step in range(n_steps):
        batch_idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_idx, :]
        Y_batch = Y[batch_idx, :]

        loss_train.append(model.loss(X, Y).detach().numpy())
        loss_test.append(model.loss(X_test, Y_test).detach().numpy())

        model.step(X_batch, Y_batch, learning_rate)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_train, label='Train')
    ax.plot(loss_test, label='Test')
    ax.legend()

    print(f'Train {loss_train[-1]}')
    print(f'Test {loss_test[-1]}')


solve_linear_sgd_torch()

# %%
A = np.array([[1, 2], [3, 4]])
A**2
A.T @ A