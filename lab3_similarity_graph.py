import numpy as np

from lab2_generation import normalize

EXPECT_NULL_CORRELATION = False


def compute_movie_similarity_graph(movie_ratings):
    n_movies = movie_ratings.shape[1]

    def users_that_rated_movie(movie):
        return np.argwhere(movie_ratings[:, movie] != 0)

    correlation = np.zeros((n_movies, n_movies))
    for movie_l in range(n_movies):
        users_l = users_that_rated_movie(movie_l)
        for movie_m in range(movie_l, n_movies):
            users_m = users_that_rated_movie(movie_m)
            users = np.intersect1d(users_l, users_m, assume_unique=True)
            if users.shape[0] == 0:
                continue
            ratings = movie_ratings[users.reshape(-1, 1),
                                    np.array([movie_l, movie_m]).reshape(1, -1)]
            mean_score = np.mean(ratings, axis=0)
            correlation[(movie_l, movie_m),
                        (movie_m, movie_l)] = np.mean(np.product(ratings - mean_score, axis=1))

    if EXPECT_NULL_CORRELATION:
        zero_auto_correlation_mask = correlation[np.arange(n_movies), np.arange(n_movies)] != 0
        new_movie_ids = np.cumsum(zero_auto_correlation_mask)
        new_movie_id_map = dict(
            zip(
                np.arange(n_movies)[zero_auto_correlation_mask],
                new_movie_ids[zero_auto_correlation_mask]))
        non_zero_correlation = correlation
        non_zero_correlation = non_zero_correlation[zero_auto_correlation_mask, :]
        non_zero_correlation = non_zero_correlation[:, zero_auto_correlation_mask]
        n_movies = non_zero_correlation.shape[0]
        correlation = non_zero_correlation

    adjacency_matrix = np.zeros((n_movies, n_movies))
    for movie_l in range(n_movies):
        for movie_m in range(movie_l + 1, n_movies):
            adjacency_matrix[(movie_l, movie_m),
                             (movie_m, movie_l)] = correlation[movie_l, movie_m] / np.sqrt(
                                 np.product(correlation[(movie_l, movie_m), (movie_l, movie_m)]))

    if EXPECT_NULL_CORRELATION:
        return adjacency_matrix, new_movie_id_map
    else:
        return adjacency_matrix


def stratify_and_normalize(adjacency_matrix):
    lowest_weights_ind = np.argsort(
        adjacency_matrix,
        axis=1,
    )[:, :-40]

    adjacency_matrix[np.arange(lowest_weights_ind.shape[0]).reshape(-1, 1), lowest_weights_ind] = 0

    adjacency_matrix = normalize(adjacency_matrix)

    return adjacency_matrix
