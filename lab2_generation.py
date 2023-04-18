# %%
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from lab2_graph_operations import diffuse_signal
from utils import random


# %%
def generate_stochastic_block_model_graph(n_nodes, n_communities, size_communities,
                                          intra_community_probability,
                                          inter_community_probability):
    accumulated_size_communities = np.cumsum(size_communities)
    adjacency_matrix = np.zeros((n_nodes, n_nodes))

    def find_community(node):
        for i in range(n_communities):
            if node < accumulated_size_communities[i]:
                return i - 1

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            community_i = find_community(i)
            community_j = find_community(j)
            if community_i == community_j:
                is_link = random.uniform() < intra_community_probability
            else:
                is_link = random.uniform() < inter_community_probability
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1 if is_link else 0

    eigenvalues, _ = np.linalg.eig(adjacency_matrix)
    adjacency_matrix /= np.abs(eigenvalues).max()

    g = nx.from_numpy_array(adjacency_matrix)
    nx.draw(g, nx.nx_agraph.graphviz_layout(g), node_color=[find_community(n) for n in g.nodes])
    return adjacency_matrix


# %%
def generate_source_localization_samples(n_nodes, n_samples, n_sources, source_value_min,
                                         source_value_max):
    # \calS
    sources = np.concatenate([
        random.choice(n_nodes, n_sources, replace=False).reshape(1, -1) for _ in range(n_samples)
    ])
    signal = np.zeros((n_samples, n_nodes))
    sources_idx = (np.arange(n_samples).reshape(-1, 1), sources)
    signal_values = random.uniform(source_value_min, source_value_max, n_samples).reshape(-1, 1)
    signal[sources_idx] = signal_values
    return signal


# %%
def generate_dataset():
    n_nodes = 50
    n_communities = 5
    intra_community_probability = 0.6
    inter_community_probability = 0.2
    size_communities = [n_nodes / n_communities] * n_communities
    adjacency_matrix = generate_stochastic_block_model_graph(n_nodes, n_communities,
                                                             size_communities,
                                                             intra_community_probability,
                                                             inter_community_probability)

    n_nodes = adjacency_matrix.shape[0]
    n_samples = 2100
    n_sources = 10  # M
    source_value_min = 0  # a
    source_value_max = 10  # b
    signal = generate_source_localization_samples(n_nodes, n_samples, n_sources, source_value_min,
                                                  source_value_max)

    n_diffusion_steps = 4
    noise_mean = 0
    noise_covariance = np.identity(1) * 1e-3
    diffused_signal = diffuse_signal(signal, adjacency_matrix, n_diffusion_steps, noise_mean,
                                     noise_covariance)

    X_train, X_test, Y_train, Y_test = train_test_split(diffused_signal,
                                                        signal,
                                                        test_size=100,
                                                        random_state=random.integers(
                                                            np.iinfo(np.int32).max),
                                                        shuffle=True)

    return adjacency_matrix, X_train, X_test, Y_train, Y_test


generate_dataset()

# %%
if __name__ == '__main__':
    n_nodes = 5
    n_communities = 2
    intra_community_probability = 0.6
    inter_community_probability = 0.2
    size_communities = [n_nodes / n_communities] * n_communities
    graph_shift_operator = generate_stochastic_block_model_graph(n_nodes, n_communities,
                                                                 size_communities,
                                                                 intra_community_probability,
                                                                 inter_community_probability)

    n_nodes = graph_shift_operator.shape[0]
    n_samples = 2100
    n_sources = 1  # M
    source_value_min = 0  # a
    source_value_max = 10  # b
    signal = generate_source_localization_samples(n_nodes, n_samples, n_sources, source_value_min,
                                                  source_value_max)

    n_diffusion_steps = 4
    noise_mean = 0
    noise_covariance = np.identity(1) * 1e-3
    diffused_signal = diffuse_signal(signal, graph_shift_operator, n_diffusion_steps, noise_mean,
                                     noise_covariance)

    X_train, X_test, Y_train, Y_test = train_test_split(diffused_signal,
                                                        signal,
                                                        test_size=100,
                                                        random_state=random.integers(
                                                            np.iinfo(np.int32).max),
                                                        shuffle=True)
