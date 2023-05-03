# %%
import numpy as np
from lab3_dataset import load_movie_dataset
from lab3_similarity_graph import compute_movie_similarity_graph, stratify_and_normalize

from utils import random

contact_index = 257

movie_ratings, movie_id_map_loading = load_movie_dataset()
contact_index = movie_id_map_loading[contact_index]
n_users = movie_ratings.shape[0]

n_train = int(np.floor(.9 * n_users))
random_indexes = np.arange(n_users)
random.shuffle(random_indexes)
train_users, test_users = random_indexes[:n_train], random_indexes[n_train:]

adjacency_matrix = compute_movie_similarity_graph(movie_ratings[train_users])
print(f'There are {adjacency_matrix.shape[0]} movies in the similarity graph')

adjacency_matrix = stratify_and_normalize(adjacency_matrix)
