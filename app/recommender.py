"""
Serving layer for ALS recommendations.
Loads pre-trained model artifacts and provides recommendation methods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

ART = Path("artifacts")


class ALSRecommender:
    """ALS-based movie recommender that loads pre-trained artifacts."""

    def __init__(self):
        """Load model artifacts and validate they exist."""
        self._validate_artifacts()

        # Load factors
        data = np.load(ART / "als.npz")
        self.user_f = data["user_factors"]
        self.item_f = data["item_factors"]

        # Load ID arrays
        self.users = np.load(ART / "users.npy")  # index → userId
        self.items = np.load(ART / "items.npy")  # index → movieId

        # Load movie titles
        movie_map = pd.read_csv(ART / "movie_map.csv")
        movie_map["movieId"] = movie_map["movieId"].astype(int)
        self.title_by_id = dict(zip(movie_map.movieId, movie_map.title))

        # Load optional popularity stats (fallbacks if missing)
        ratings_path = Path("data_small/ratings.csv")
        if ratings_path.exists():
            ratings = pd.read_csv(ratings_path, usecols=["movieId"])
            counts = ratings.groupby("movieId").size()
            self.popularity = counts.to_dict()
            self._pop_max = float(counts.max()) if not counts.empty else 1.0
        else:
            self.popularity = {}
            self._pop_max = 1.0

        # Reconstruct model object (just attach factors)
        self.model = AlternatingLeastSquares()
        self.model.user_factors = self.user_f
        self.model.item_factors = self.item_f

        # Load serving matrix (users × items)
        self.X_users_items = sp.load_npz(ART / "X_users_items.npz").tocsr()

        # Build ID → index lookup dicts
        self.uidx = {int(u): i for i, u in enumerate(self.users)}
        self.iidx = {int(m): i for i, m in enumerate(self.items)}

    def _validate_artifacts(self):
        """Check that all required artifacts exist."""
        if not ART.exists():
            raise FileNotFoundError(
                f"Artifacts directory '{ART}' not found. "
                "Run 'python train_als.py' first."
            )

        required = [
            "als.npz",
            "users.npy",
            "items.npy",
            "X_users_items.npz",
            "movie_map.csv",
        ]
        missing = [f for f in required if not (ART / f).exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing artifacts: {missing}. Run 'python train_als.py' first."
            )

    def recommend_known(self, user_id: int, k: int = 10, novelty: float = 0.0):
        """
        Recommend movies for a known user.

        Args:
            user_id: User ID from training data
            k: Number of recommendations
            novelty: Optional novelty weight in [0, 1]

        Returns:
            List of dicts with movieId and title
        """
        if user_id not in self.uidx:
            return []

        u = self.uidx[user_id]
        user_row = self.X_users_items[u : u + 1]  # Extract user's interaction row

        # Get recommendations (filtering already-liked items)
        fetch_k = k if novelty <= 0 else max(k * 3, k)
        recs = self.model.recommend(
            u,
            user_row,
            N=fetch_k,
            filter_already_liked_items=True,
            recalculate_user=False,
        )

        formatted = self._format_results(recs)
        if novelty > 0:
            formatted = self._apply_novelty(formatted, novelty)
        return formatted[:k]

    def recommend_cold(self, liked_movie_ids=None, k: int = 10, novelty: float = 0.0):
        """
        Recommend movies for cold-start (new user with seed likes).

        Args:
            liked_movie_ids: List of movie IDs the user likes
            k: Number of recommendations
            novelty: Optional novelty weight in [0, 1]

        Returns:
            List of dicts with movieId and title
        """
        liked_movie_ids = liked_movie_ids or []

        # Build synthetic user row from seed movies
        row = sp.csr_matrix((1, len(self.items)), dtype=np.float32)

        if liked_movie_ids:
            # Map movie IDs to indices (skip unknown movies)
            cols = [self.iidx[m] for m in liked_movie_ids if m in self.iidx]

            if cols:
                # Use same confidence as training (1 + alpha = 21)
                data = np.full(len(cols), 21.0, dtype=np.float32)
                row = sp.csr_matrix(
                    (data, ([0] * len(cols), cols)), shape=(1, len(self.items))
                )

        # Manual cold-start recommendation:
        # 1. Compute user embedding from seed movies
        # 2. Score all items
        # 3. Filter and rank

        fetch_k = k if novelty <= 0 else max(k * 3, k) + len(liked_movie_ids)

        if not cols:
            # No valid seed movies - return popular items
            # Simple fallback: return first fetch_k items
            indices = np.arange(min(fetch_k, len(self.items)))
            scores = np.ones(len(indices))
            recs = (indices, scores)
        else:
            # Compute user embedding: solve for u_new using seed items
            # u_new = (I^T @ I + lambda*I)^-1 @ I^T @ r
            # where I = item factors for seed movies, r = confidence vector
            seed_indices = cols
            item_vecs = self.item_f[seed_indices]  # (n_seeds × factors)
            confidence = np.full(len(seed_indices), 21.0)

            # Solve weighted least squares
            A = item_vecs.T @ (item_vecs * confidence[:, None])
            A += 0.01 * np.eye(A.shape[0])  # Regularization
            b = item_vecs.T @ confidence
            user_vec = np.linalg.solve(A, b)

            # Score all items
            scores = self.item_f @ user_vec

            # Filter seed movies
            scores[seed_indices] = -np.inf

            # Get top-k
            top_indices = np.argpartition(scores, -fetch_k)[-fetch_k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            recs = (top_indices, scores[top_indices])

        # Format results
        formatted = self._format_results(recs)

        # Additional filtering
        seed_set = set(liked_movie_ids)
        formatted = [item for item in formatted if item["movieId"] not in seed_set]

        if novelty > 0:
            formatted = self._apply_novelty(formatted, novelty)

        return formatted[:k]

    def _format_results(self, recs):
        formatted = []
        # model.recommend() returns (indices_array, scores_array) tuple
        if isinstance(recs, tuple) and len(recs) == 2:
            indices, scores = recs
            for idx, score in zip(indices, scores):
                mid = int(self.items[idx])
                formatted.append(
                    {
                        "movieId": mid,
                        "title": self.title_by_id.get(mid, str(mid)),
                        "score": float(score),
                        "raw_score": float(score),
                    }
                )
        else:
            # Legacy format: list of (idx, score) tuples
            for idx, score in recs:
                mid = int(self.items[idx])
                formatted.append(
                    {
                        "movieId": mid,
                        "title": self.title_by_id.get(mid, str(mid)),
                        "score": float(score),
                        "raw_score": float(score),
                    }
                )
        return formatted

    def _apply_novelty(self, recs, novelty):
        if not self.popularity or novelty <= 0:
            return recs

        for item in recs:
            pop = self.popularity.get(item["movieId"], 0.0)
            penalty = novelty * (pop / self._pop_max)
            item["score"] = float(item["raw_score"] - penalty)

        return sorted(recs, key=lambda x: x["score"], reverse=True)
