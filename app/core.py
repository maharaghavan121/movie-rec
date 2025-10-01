"""
Shared data loading and matrix building functions.
Used by both training (train_als.py) and serving (app/recommender.py).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data():
    """Load MovieLens CSV files."""
    movies = pd.read_csv("data_small/movies.csv")
    ratings = pd.read_csv("data_small/ratings.csv")
    links = pd.read_csv("data_small/links.csv")
    tags = pd.read_csv("data_small/tags.csv")
    return movies, ratings, links, tags


def make_splits(ratings, thresh=4):
    """
    Time-based train/val/test split.

    For each user with ≥2 positive ratings (rating ≥ thresh):
    - Test: last positive rating
    - Val: second-to-last positive rating
    - Train: all earlier positive ratings

    Args:
        ratings: DataFrame with userId, movieId, rating, timestamp
        thresh: Minimum rating to consider positive (default: 4.0)

    Returns:
        train, val, test DataFrames
    """
    df = ratings.sort_values(["userId", "timestamp"]).copy()
    df["is_positive_rating"] = df["rating"] >= thresh

    pos = df[df["is_positive_rating"]].copy()
    pos["rnk"] = pos.groupby("userId")["timestamp"].rank(method="first")
    pos["num_positives"] = pos.groupby("userId")["userId"].transform("size")
    pos = pos[pos["num_positives"] >= 2].copy()

    test = pos[pos["rnk"] == pos["num_positives"]].copy()
    val = pos[pos["rnk"] == pos["num_positives"] - 1].copy()
    train = pos[pos["rnk"] < pos["num_positives"] - 1].copy()

    return train, val, test


def build_index_maps(ratings, train_data):
    """
    Build user and movie ID to index mappings.

    Args:
        ratings: Full ratings DataFrame (unused, kept for compatibility)
        train_data: Training DataFrame

    Returns:
        user_ids: Sorted array of user IDs
        movie_ids: Sorted array of movie IDs
        user_index_map: Dict mapping user_id → index
        movie_index_map: Dict mapping movie_id → index
    """
    user_ids = np.sort(train_data["userId"].unique())
    movie_ids = np.sort(train_data["movieId"].unique())
    user_index_map = {u: i for i, u in enumerate(user_ids)}
    movie_index_map = {m: i for i, m in enumerate(movie_ids)}
    return user_ids, movie_ids, user_index_map, movie_index_map


def train_matrix(
    train_data, user_ids, movie_ids, user_index_map, movie_index_map, alpha=20.0
):
    """
    Build sparse user-item interaction matrix.

    Matrix orientation: users × items (serving orientation)
    Confidence weighting: 1 + alpha for positive interactions

    Args:
        train_data: Training DataFrame with userId, movieId
        user_ids: Array of user IDs
        movie_ids: Array of movie IDs
        user_index_map: user_id → index mapping
        movie_index_map: movie_id → index mapping
        alpha: Confidence parameter (default: 20.0)

    Returns:
        Sparse CSR matrix of shape (n_users, n_items)
    """
    rows = train_data["userId"].map(user_index_map).to_numpy()
    cols = train_data["movieId"].map(movie_index_map).to_numpy()
    data = np.full(len(train_data), 1.0 + alpha, dtype="float32")

    X = sp.coo_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(movie_ids)), dtype="float32"
    ).tocsr()

    return X
