# %%
import numpy as np

from lab3_dataset import (create_samples_from_ratings, load_movie_dataset,
                          remove_users_not_rating_movie, split_train_test_users)
from lab3_similarity_graph import (compute_movie_similarity_graph, stratify_and_normalize)
from utils import random

# %%
contact_index = 257

movie_ratings, movie_id_map_loading = load_movie_dataset()
contact_index = movie_id_map_loading[contact_index]
n_users = movie_ratings.shape[0]

users_train, users_test = split_train_test_users(n_users)

adjacency_matrix = compute_movie_similarity_graph(movie_ratings[users_train])
print(f'There are {adjacency_matrix.shape[0]} movies in the similarity graph')

adjacency_matrix = stratify_and_normalize(adjacency_matrix)

# %%
movie_ratings_train = remove_users_not_rating_movie(movie_ratings[users_train], contact_index)
movie_ratings_test = remove_users_not_rating_movie(movie_ratings[users_test], contact_index)

print(f'{movie_ratings_train.shape[0]} samples for training')
print(f'{movie_ratings_test.shape[0]} samples for testing')

# %%

X_train, Y_train = create_samples_from_ratings(movie_ratings_train, contact_index)
X_test, Y_test = create_samples_from_ratings(movie_ratings_test, contact_index)
