"""Streamlit UI for exploring the ALS recommender."""

from pathlib import Path
from typing import List
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when invoked via `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.recommender import ALSRecommender


@st.cache_resource(show_spinner=False)
def load_recommender() -> ALSRecommender:
    return ALSRecommender()


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    path = Path("data_small/movies.csv")
    if not path.exists():
        raise FileNotFoundError(
            "data_small/movies.csv not found. Download MovieLens and place it under data_small/."
        )
    df = pd.read_csv(path)
    df["movieId"] = df["movieId"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    path = Path("data_small/ratings.csv")
    if not path.exists():
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    df = pd.read_csv(path)
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    return df


def _format_results(rows: List[dict], movies: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["movieId", "title", "score", "raw_score"])
    df = pd.DataFrame(rows)
    return df.merge(movies[["movieId", "genres"]], on="movieId", how="left")


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommender: Personalization vs Novelty")

    try:
        recommender = load_recommender()
        movies = load_movies()
        ratings = load_ratings()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Controls")
    k = st.sidebar.slider("Top K", min_value=5, max_value=30, value=10, step=1)
    novelty = st.sidebar.slider(
        "Novelty weight",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
        help="Higher values penalize globally popular titles to highlight long-tail movies.",
    )

    tab_known, tab_cold = st.tabs(["Known Users", "Cold-start"])

    with tab_known:
        st.subheader("Personalized for an existing user")
        user_options = [int(u) for u in recommender.users]
        if not user_options:
            st.warning("No users found in artifacts. Train the model first.")
            st.stop()
        user_id = st.selectbox("Select a userId", options=user_options, index=0)

        liked = pd.DataFrame()
        if not ratings.empty:
            liked = (
                ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4.0)]
                .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
                .sort_values(["rating", "timestamp"], ascending=[False, False])
                .head(5)
            )

        if novelty > 0:
            st.caption("Novelty is on — reranking by penalizing popularity.")

        results = recommender.recommend_known(user_id=user_id, k=k, novelty=novelty)
        rec_df = _format_results(results, movies)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Recommendations")
            if rec_df.empty:
                st.info("No recommendations available for this user.")
            else:
                display_cols = ["movieId", "title", "genres", "score", "raw_score"]
                st.dataframe(rec_df[display_cols].reset_index(drop=True))
        with col2:
            st.write("### Recent Likes (rating ≥ 4)")
            if liked.empty:
                st.info("No positive history for this user in ratings.csv.")
            else:
                st.dataframe(
                    liked[["movieId", "title", "genres", "rating"]].reset_index(
                        drop=True
                    )
                )

    with tab_cold:
        st.subheader("New user with a few favorites")
        search_movies = movies.sort_values("title")
        default_choices = search_movies.head(5)["title"].tolist()
        selections = st.multiselect(
            "Pick a few favorites",
            options=search_movies["title"].tolist(),
            default=default_choices[:2],
        )

        seed_ids = (
            search_movies[search_movies["title"].isin(selections)]["movieId"]
            .astype(int)
            .tolist()
            if selections
            else []
        )

        if seed_ids:
            st.caption(f"Seeding with movieIds: {seed_ids}")
        else:
            st.caption("Select at least one movie to see cold-start recommendations.")

        if seed_ids:
            cold_results = recommender.recommend_cold(
                liked_movie_ids=seed_ids, k=k, novelty=novelty
            )
            cold_df = _format_results(cold_results, movies)
            if cold_df.empty:
                st.info(
                    "No cold-start recommendations available for the selected seeds."
                )
            else:
                st.dataframe(
                    cold_df[
                        ["movieId", "title", "genres", "score", "raw_score"]
                    ].reset_index(drop=True)
                )

    st.sidebar.write("---")
    st.sidebar.markdown(
        "**Tip:** Run `python train_als.py` whenever you tweak hyperparameters to refresh the artifacts."
    )


if __name__ == "__main__":
    main()
