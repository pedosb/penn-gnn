import numpy as np

from utils import random


def diffuse_signal(signal: np.ndarray, graph_shift_operator, n_diffusion_steps, noise_mean,
                   noise_covariance):
    for _ in range(n_diffusion_steps):
        noise = random.normal(noise_mean, np.sqrt(noise_covariance), size=signal.shape)
        diffused_signal = (graph_shift_operator @ signal.T).T + noise
        signal = diffused_signal
    return diffused_signal


def filter_graph_signal(filter_coefficients, graph_shift_operator, signal):
    """Apply the graph filter

    Args:
        filter_coefficients (tensor): order x n_features_in x n_features_out
        graph_shift_operator (tensor): n_nodes x n_nodes
        signal (tensor): n_samples x n_nodes x n_features

    Returns:
        tensor: n_samples x n_nodes x n_features_out
    """
    signal_step = signal
    filtered_signal = signal_step @ filter_coefficients[0]
    for coefficient in filter_coefficients[1:]:
        signal_step = (graph_shift_operator @ signal_step)
        filtered_signal += signal_step @ coefficient
    return filtered_signal
