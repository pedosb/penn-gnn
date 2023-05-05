import zipfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from lab3_similarity_graph import (compute_movie_similarity_graph, stratify_and_normalize)
from utils import random


def load_movie_dataset():
    with zipfile.ZipFile('data/ml-100k.zip') as zipf:
        u_file = zipf.open('ml-100k/u.data')
        u_data = np.loadtxt(u_file, np.uint32, delimiter='\t', usecols=(0, 1, 2))

    n_users = np.unique(u_data[:, 0]).shape[0]
    n_movies = np.unique(u_data[:, 1]).shape[0]

    movie_ratings = np.zeros((
        n_users,
        n_movies,
    ), np.uint8)

    movie_ratings[u_data[:, 0] - 1, u_data[:, 1] - 1] = u_data[:, 2]

    selected_movies_mask = np.sum(movie_ratings > 0, axis=0) >= 150
    new_movie_ids = np.cumsum(selected_movies_mask) - 1
    new_movie_id_map = dict(
        zip(np.arange(n_movies)[selected_movies_mask], new_movie_ids[selected_movies_mask]))
    cleaned_movie_ratings = movie_ratings[:, selected_movies_mask]

    return cleaned_movie_ratings, new_movie_id_map


def split_users(n_users, ratio_train=0.9, ratio_validation=0.09, n_sets=1):
    n_test = int(np.floor((1 - ratio_train) * n_users))
    n_validation = int(np.floor(ratio_validation * n_users))
    random_indexes = np.arange(n_users)
    random.shuffle(random_indexes)
    random_indexes = random_indexes[:n_test * n_sets].reshape(-1, n_test)
    indexes = np.arange(n_users)
    for i in range(n_sets):
        users_test = random_indexes[i]
        users_train, users_validation = train_test_split(np.setdiff1d(indexes, users_test),
                                                         test_size=n_validation)
        print(n_users, n_users - n_validation - n_test, n_validation, n_test)
        yield users_train, users_validation, users_test


def remove_users_not_rating_movie(movie_ratings, movie):
    return movie_ratings[movie_ratings[:, movie] != 0]


def create_samples_from_ratings(movie_ratings, target_movie):
    X = movie_ratings.copy()
    X[:, target_movie] = 0
    Y = np.zeros_like(X)
    Y[:, target_movie] = movie_ratings[:, target_movie]
    X = np.expand_dims(X, -1)
    return X, Y


def generate_dataset(users_train, users_validation, users_test, target_movie):
    adjacency_matrix = compute_movie_similarity_graph(movie_ratings[users_train])
    if verbose:
        print(f'There are {adjacency_matrix.shape[0]} movies in the similarity graph')

    adjacency_matrix = stratify_and_normalize(adjacency_matrix)

    movie_ratings_train = remove_users_not_rating_movie(movie_ratings[users_train], target_movie)
    movie_ratings_validation = remove_users_not_rating_movie(movie_ratings[users_validation],
                                                             target_movie)
    movie_ratings_test = remove_users_not_rating_movie(movie_ratings[users_test], target_movie)

    if verbose:
        print(f'{movie_ratings_train.shape[0]} samples for training')
        print(f'{movie_ratings_test.shape[0]} samples for testing')

    X_train, Y_train = create_samples_from_ratings(movie_ratings_train, target_movie)
    X_validation, Y_validation = create_samples_from_ratings(movie_ratings_validation,
                                                             target_movie)
    X_test, Y_test = create_samples_from_ratings(movie_ratings_test, target_movie)

    adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test, Y_validation = [
        torch.tensor(v, dtype=torch.float32)
        for v in [adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test, Y_validation]
    ]
    return adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test, Y_validation


if __name__ == '__main__':
    verbose = False

    contact_index = 257
    movie_ratings, movie_id_map_loading = load_movie_dataset()
    contact_index = movie_id_map_loading[contact_index]
    n_users = movie_ratings.shape[0]

    for i, (users_train, users_validation, users_test) in enumerate(
            split_users(n_users, ratio_train=.9, ratio_validation=0.09, n_sets=8)):
        adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test, Y_validation = generate_dataset(
            users_train, users_validation, users_test, contact_index)

        torch.save([
            contact_index, adjacency_matrix, X_train, X_test, X_validation, Y_train, Y_test,
            Y_validation
        ], f'datasets/lab3_similarity_{i:02d}.pt')
