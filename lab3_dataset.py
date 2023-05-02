import numpy as np
import zipfile


def load_movie_dataset():
    with zipfile.ZipFile('data/ml-100k.zip') as zipf:
        u_file = zipf.open('ml-100k/u.data')
        u_data = np.loadtxt(u_file, np.uint32, delimiter='\t', usecols=(0, 1, 2))

    n_users = np.unique(u_data[:, 0]).shape[0]
    n_movies = np.unique(u_data[:, 1]).shape[0]

    movie_ratings = np.empty((
        n_users,
        n_movies,
    ), np.uint8)

    movie_ratings[u_data[:, 0] - 1, u_data[:, 1] - 1] = u_data[:, 2]

    contact_index = 257
    selected_movies_mask = np.sum(movie_ratings > 0, axis=0) >= 150
    new_movie_id_mapping = np.cumsum(selected_movies_mask)
    contact_index = new_movie_id_mapping[contact_index]
    cleaned_movie_ratings = movie_ratings[:, selected_movies_mask]

    return cleaned_movie_ratings
