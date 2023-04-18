import numpy as np

from utils import random


def diffuse_signal(signal: np.ndarray, graph_shift_operator, n_diffusion_steps, noise_mean,
                   noise_covariance):
    for _ in range(n_diffusion_steps):
        noise = random.normal(noise_mean, np.sqrt(noise_covariance), size=signal.shape)
        diffused_signal = (graph_shift_operator @ signal.T).T + noise
        signal = diffused_signal
    return diffused_signal


def filter(filter_coefficients, signal, graph_shift_operator):
    filtered_signal = None
    for shift, coefficient in enumerate(filter_coefficients):
        signal_step = ((graph_shift_operator**shift) @ signal.T).T * coefficient
        if filtered_signal is None:
            filtered_signal = signal_step
        else:
            filtered_signal += signal_step
    return filtered_signal
