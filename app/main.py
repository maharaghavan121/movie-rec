"""FastAPI surface for the ALS recommender."""

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from app.recommender import ALSRecommender

app = FastAPI(title="Movie Recommender", version="0.1.0")

_recommender: Optional[ALSRecommender] = None
_load_error: Optional[str] = None


def _get_recommender() -> ALSRecommender:
    global _recommender, _load_error
    if _recommender is None:
        try:
            _recommender = ALSRecommender()
        except FileNotFoundError as exc:  # Artifacts not generated yet
            _load_error = str(exc)
            raise HTTPException(status_code=503, detail=_load_error)
    return _recommender


@app.get("/health")
def health():
    """Return basic service status."""
    if _recommender is None and _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)

    try:
        rec = _get_recommender()
    except HTTPException as exc:
        # Surface artifact issues via health endpoint
        raise exc

    return {
        "status": "ok",
        "model": "ALS",
        "n_users": int(rec.user_f.shape[0]),
        "n_items": int(rec.item_f.shape[0]),
    }


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="MovieLens userId"),
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):
    """Recommend movies for an existing user."""
    rec = _get_recommender()
    recs = rec.recommend_known(user_id=user_id, k=k)
    if not recs:
        raise HTTPException(
            status_code=404, detail=f"User {user_id} not found in training data"
        )
    return {"user_id": user_id, "k": k, "results": recs}


@app.get("/recommend_cold")
def recommend_cold(
    liked: Optional[str] = Query(None, description="Comma-separated seed movieIds"),
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):
    """Recommend movies for a cold-start scenario using seed movie IDs."""
    rec = _get_recommender()

    liked_ids: List[int] = []
    if liked:
        try:
            liked_ids = [int(x.strip()) for x in liked.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid liked parameter: {exc}"
            )

    results = rec.recommend_cold(liked_ids, k=k)
    return {"liked": liked_ids, "k": k, "results": results}
