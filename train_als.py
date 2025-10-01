"""
Train ALS model and save artifacts for serving.

Usage:
    python train_als.py

Outputs:
    artifacts/als.npz - User and item factors
    artifacts/users.npy - User ID array
    artifacts/items.npy - Movie ID array
    artifacts/X_users_items.npz - Serving matrix (users × items)
    artifacts/movie_map.csv - Movie titles
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from app.core import load_data, make_splits, build_index_maps, train_matrix

# Artifact directory
ART = Path("artifacts")
ART.mkdir(exist_ok=True, parents=True)


def _ensure_pairs(recs):
    """Normalize implicit's recommend output to list of (idx, score)."""
    if isinstance(recs, tuple) and len(recs) == 2 and not isinstance(recs[0], tuple):
        ids, scores = recs
        return list(zip(ids, scores))
    return list(recs)


def _recommended_movie_ids(model, X, user_idx, movie_ids, k):
    # Manual scoring to avoid transposed factor issues
    # Model was trained on X.T, so:
    # - model.user_factors = item factors (n_items × n_factors)
    # - model.item_factors = user factors (n_users × n_factors)

    user_row = X[user_idx : user_idx + 1]
    user_vec = model.item_factors[user_idx]  # Get user's latent vector
    scores = model.user_factors @ user_vec  # Compute scores for all items

    # Filter already-liked items
    liked_items = set(user_row.indices)
    scores[list(liked_items)] = -np.inf

    # Get top-k
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    return [int(movie_ids[i]) for i in top_indices]


def _metric_over_test(test_data, user_index_map, movie_index_map, fn):
    hits = []
    for _, row in test_data.iterrows():
        u_id, m_id = row["userId"], row["movieId"]
        u_idx = user_index_map.get(u_id)
        m_idx = movie_index_map.get(m_id)
        # Skip if user or movie not in training set
        if u_idx is None or m_idx is None:
            continue
        hits.append(fn(u_idx, m_id))
    return float(np.mean(hits)) if hits else 0.0


def recall_at_k(model, X, movie_ids, user_index_map, movie_index_map, test_data, k=10):
    """Recall@K with one relevant item per user."""

    def _fn(idx, target):
        preds = _recommended_movie_ids(model, X, idx, movie_ids, k)
        return 1.0 if target in preds else 0.0

    return _metric_over_test(test_data, user_index_map, movie_index_map, _fn)


def precision_at_k(
    model, X, movie_ids, user_index_map, movie_index_map, test_data, k=10
):
    """Precision@K reduces to recall/k with leave-one-out setup."""
    return recall_at_k(
        model, X, movie_ids, user_index_map, movie_index_map, test_data, k
    ) / float(k)


def ndcg_at_k(model, X, movie_ids, user_index_map, movie_index_map, test_data, k=10):
    """NDCG@K considers the rank of the held-out item."""

    def _fn(idx, target):
        preds = _recommended_movie_ids(model, X, idx, movie_ids, k)
        if target not in preds:
            return 0.0
        rank = preds.index(target) + 1
        return 1.0 / np.log2(rank + 1)

    return _metric_over_test(test_data, user_index_map, movie_index_map, _fn)


@dataclass
class TrainingConfig:
    factors: int = 64
    regularization: float = 0.02
    iterations: int = 20
    alpha: float = 20.0


def main():
    print("=" * 70)
    print("Training ALS Model")
    print("=" * 70)

    config = TrainingConfig()

    # Load data
    print("\n[1/6] Loading data...")
    movies, ratings, links, tags = load_data()
    print(f"  ✓ Loaded {len(ratings):,} ratings, {len(movies):,} movies")

    # Build splits
    print("\n[2/6] Building train/val/test splits...")
    train_data, val_data, test_data = make_splits(ratings, thresh=4)
    print(f"  ✓ Train: {len(train_data):,} ratings")
    print(f"  ✓ Val:   {len(val_data):,} ratings")
    print(f"  ✓ Test:  {len(test_data):,} ratings")

    # Build index maps
    print("\n[3/6] Building index maps...")
    user_ids, movie_ids, user_index_map, movie_index_map = build_index_maps(
        ratings, train_data
    )
    print(f"  ✓ Users:  {len(user_ids):,}")
    print(f"  ✓ Movies: {len(movie_ids):,}")

    # Build matrix
    print("\n[4/6] Building user-item matrix...")
    X_users_items = train_matrix(
        train_data,
        user_ids,
        movie_ids,
        user_index_map,
        movie_index_map,
        alpha=config.alpha,
    )
    print(f"  ✓ Shape: {X_users_items.shape}")
    density = (
        100 * X_users_items.nnz / (X_users_items.shape[0] * X_users_items.shape[1])
    )
    print(f"  ✓ Density: {density:.2f}% ({X_users_items.nnz:,} non-zero entries)")

    # Train ALS
    print(
        "\n[5/6] Training ALS (factors={config.factors}, reg={config.regularization}, iters={config.iterations})..."
    )
    als = AlternatingLeastSquares(
        factors=config.factors,
        regularization=config.regularization,
        iterations=config.iterations,
        random_state=42,
    )
    # Transpose once here: implicit expects items × users
    als.fit(X_users_items.T)
    print("  ✓ Training complete")

    # Evaluate
    print("\n[6/6] Evaluating on test set...")
    metrics = {}
    for k in (5, 10, 20):
        metrics[f"Recall@{k}"] = recall_at_k(
            als, X_users_items, movie_ids, user_index_map, movie_index_map, test_data, k
        )
        metrics[f"Precision@{k}"] = precision_at_k(
            als, X_users_items, movie_ids, user_index_map, movie_index_map, test_data, k
        )
        metrics[f"NDCG@{k}"] = ndcg_at_k(
            als, X_users_items, movie_ids, user_index_map, movie_index_map, test_data, k
        )

    for key, value in metrics.items():
        print(f"  ✓ {key}: {value:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")
    # Model was trained on X.T, so factors are swapped - swap back when saving
    np.savez_compressed(
        ART / "als.npz",
        user_factors=als.item_factors.astype(
            np.float32
        ),  # Swapped: these are user factors
        item_factors=als.user_factors.astype(
            np.float32
        ),  # Swapped: these are item factors
    )
    print("  ✓ als.npz")

    np.save(ART / "users.npy", user_ids)
    print("  ✓ users.npy")

    np.save(ART / "items.npy", movie_ids)
    print("  ✓ items.npy")

    sp.save_npz(ART / "X_users_items.npz", X_users_items)
    print("  ✓ X_users_items.npz")

    movies[["movieId", "title"]].to_csv(ART / "movie_map.csv", index=False)
    print("  ✓ movie_map.csv")

    with open(ART / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("  ✓ metrics.json")

    with open(ART / "config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)
    print("  ✓ config.json")

    print("\n" + "=" * 70)
    print("✓ Training complete! Artifacts saved to artifacts/")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start API: uvicorn app.main:app --reload")
    print("  2. Test: curl 'http://127.0.0.1:8000/recommend?user_id=1&k=5'")
    print()


if __name__ == "__main__":
    main()
