# movie-recs

Awesome pick—**Movie + Snack Night** is a perfect (fun!) way to learn real recommender-system skills end-to-end.

Below is a **tight, weekend-scopable plan** that (a) avoids the flaky HF dataset you saw, (b) uses robust, public sources, and (c) teaches retrieval → ranking → re-ranking with a **hybrid CF + content** stack, plus the “snack pairing” side quest.

## Data Setup

This repo keeps the training data out of version control. Download the MovieLens `ml-latest-small` archive from [GroupLens](https://grouplens.org/datasets/movielens/latest/) and unpack it into a `data_small/` directory at the project root (`ratings.csv`, `movies.csv`, etc.). All scripts assume that folder exists locally but it should stay ignored by git.

For a quick look at the expected schema, peek at the tiny CSVs in `sample_data/`. They are just illustrative slices and are not suitable for training.

---

## Rock‑solid datasets (no HF needed)

**Movies (interactions + IDs)**

* **MovieLens “latest‑small”** for fast iteration: `ratings.csv`, `movies.csv`, `links.csv`. (\~100k ratings) — ideal for a laptop prototype. ([GroupLens][1])
* When you want scale, swap in **MovieLens 25M** (same schema, plus tag‑genome files). ([GroupLens][2])
* `links.csv` gives `movieId ↔ imdbId ↔ tmdbId` so you can enrich metadata from TMDB‑derived files below. ([GroupLens Files][3])

**Movies (rich content metadata)**

* **The Movies Dataset** (Kaggle; TMDB‑derived): `movies_metadata.csv`, `keywords.csv`, `credits.csv` (cast/crew), etc.—exactly what you need for plot/genre/cast text features. ([Kaggle][4])

**Recipes (for the pairing)**

* **Food.com Recipes & Interactions**: `RAW_recipes.csv` (ingredients, tags, minutes, steps) + `RAW_interactions.csv` (reviews/ratings). \~180k recipes, \~700k interactions. Great for quick “snackness” rules or a tiny recipe ranker. ([Kaggle][5])

> Optional extra plot text: **CMU Movie Summary Corpus** (42k plot summaries) if you want long, descriptive plots to embed. ([ark.cs.cmu.edu][6])

---

## What you’ll build (MVP)

> **Goal:** Given a handful of seed likes (yours + hers) and an optional “mood” (cozy/rom‑com/action etc.), recommend **10 movies** and pair each with **1 snack recipe** you can actually make.

**Pipeline (teach you the right sub‑skills):**

1. **Ingest & join**

* Load MovieLens *small* (`ratings.csv`, `movies.csv`, `links.csv`).
* Left‑join to The Movies Dataset on `tmdbId` via `links.csv` → you get `overview`, `genres`, `keywords`, top‑billed cast/crew. (Beware: `movies_metadata.id` is a string; coerce to `int`, handle bad rows.) ([GroupLens Files][3])

2. **Split for offline eval (so you can measure)**

* **Leave‑one‑out per user** (keep the most recent rating as test, second‑most as validation).
* Convert to **implicit** feedback (e.g., treat ratings ≥ 4 as positives).

3. **Baselines you should beat**

* **Most‑Popular** (by global count).
* **Item‑KNN** (co‑occurrence / cosine of item vectors). Easy and strong. Use the `implicit` library. ([Benfred][7])

4. **Collaborative Filtering (core CF)**

* Train **ALS for implicit feedback** (BM25 or TF‑IDF reweighting) with `implicit`. Export user/item factors for candidate gen. Evaluate Recall\@K / NDCG\@K. ([Benfred][7])

5. **Content tower (cold‑start & control)**

* Build text for each movie:
  `text = title + " " + overview + " " + genres + top_5_cast + director + " " + keywords`
* Embed with a strong, small model (pick one):
  **e5‑base‑v2** (great quality) or **all‑MiniLM‑L6‑v2** (fast & tiny). ([Hugging Face][8])
* For a user, average the embeddings of their top‑rated movies → **profile vector**.
* Build an ANN index (**FAISS**) over item embeddings for **candidate generation**. ([GitHub][9])

6. **Hybrid learning‑to‑rank (your “non‑trivial model”)**

* For each `(user, test‑query)` produce 200–500 candidates (union of CF + content).
* Compute **features per (user,item)**:

  * `cf_als_score`, `cf_itemknn_score`
  * `content_cosine(user_prof, item_embed)`
  * `same_genre_count`, `cast_overlap`, `director_match`, `year_gap`, `log_popularity`
* Train **LightGBM LGBMRanker** (LambdaMART with `rank:ndcg` analogue) on validation triples with user‑grouping. This teaches you production‑shaped **re‑ranking**. ([lightgbm.readthedocs.io][10])
* Report **Recall\@10 / NDCG\@10** vs. ALS and ItemKNN.

7. **Snack pairing (fun but real)**

* Turn each recipe into a text string:
  `name + " " + tags + " " + ingredients` (+ optionally minutes).
* Embed recipes with the **same** text model you used for movies.
* **Heuristic slate rules** for “snack night”: filter `minutes ≤ 25`, prefer `tags` ∈ {snack, appetizer, finger‑food, dessert}, then pick top‑1 by cosine similarity to the **movie’s** embedding (or to a “vibe” prompt like “cozy romantic evening”).
* If you want learning here too, train a tiny recipe ranker using Food.com interactions (predict 5‑star vs not) and blend that score with similarity. ([Kaggle][11])

8. **Tiny demo**

* One route API (`/recommend?user=A&user=B&mood=cozy`) that returns `[movie, why, snack] × 10`.
* Show “**why**” features (e.g., “+genre: Romance, +cast overlap: 2, +content sim: 0.61, +ALS: 1.4”).
* Optional **MMR diversity** in the last step to avoid ten near‑duplicates.

---

## Exact files & schemas you’ll touch (so it’s concrete)

**MovieLens latest‑small**

* `ratings.csv`: `userId, movieId, rating, timestamp`
* `movies.csv`: `movieId, title, genres` (pipe‑separated genres)
* `links.csv`: `movieId, imdbId, tmdbId` (for joining to TMDB metadata). ([GroupLens][1])

**The Movies Dataset**

* `movies_metadata.csv`: `id (tmdb)`, `genres` (JSON‑ish), `overview`, `release_date`, `runtime`, etc.
* `keywords.csv`: list of keyword JSONs per TMDB `id`.
* `credits.csv`: `cast` & `crew` JSON (get top 5 cast; director from crew). ([Kaggle][4])

**Food.com**

* `RAW_recipes.csv`: `name, id, minutes, tags, nutrition, n_steps, ingredients...`
* `RAW_interactions.csv`: `user_id, recipe_id, date, rating, review` (useful if you train a tiny recipe scorer). ([Kaggle][12])

---

## “Do this now” build order (scoped for a weekend)

### Phase 1 — Data & baselines (CF)

1. **Load & clean.** Coerce `movies_metadata.id` to `Int64`; drop rows that don’t parse; left‑join on `tmdbId`. (Known quirk.) ([Kaggle][4])
2. **Train ALS (implicit)** with BM25 reweighting; evaluate Recall\@10 / NDCG\@10 using leave‑one‑out. Export item factors. ([Benfred][7])
3. **ItemKNN** as another baseline. ([Benfred][7])

### Phase 2 — Content retrieval + FAISS

4. Build movie text, embed with **e5‑base‑v2** *or* **all‑MiniLM‑L6‑v2**; construct user profiles by averaging liked items. ([Hugging Face][8])
5. Index items with **FAISS** (inner product index) → fast content candidates. ([GitHub][9])

### Phase 3 — Hybrid LTR re‑ranker

6. Generate 500 candidates/user (ALS top‑300 ∪ FAISS top‑300).
7. Build the feature matrix (scores + overlaps + recency), **group by user**; train **LightGBM Ranker** (objective = Lambdarank). Validate and then test. ([lightgbm.readthedocs.io][10])
8. **Ablations:**

   * CF‑only vs Content‑only vs Hybrid
   * No‑cast vs +cast features
   * No‑mood vs +mood (encode “mood” as a small text vector and add cosine with item).

### Phase 4 — Snack pairing

9. Embed recipes; filter by `minutes ≤ 25` + snack‑like tags; pick **1** per movie by cosine to the movie embedding or to the “mood” prompt. ([Kaggle][12])
10. (Optional) Train a 2‑feature logistic or LightGBM **recipe scorer** using Food.com interactions: features could be `minutes`, `n_steps`, bag‑of‑tags, and embedding similarity to the *movie* → combine with a 70/30 blend.

---

## Evaluation you can speak to

* **Offline ranking:** **Recall\@10 / NDCG\@10** on the leave‑one‑out test set.
* **Cold‑start:** Hide popular titles at train time and test content‑only retrieval.
* **Diversity:** Report **intra‑list similarity (ILS)** and **genre coverage** before/after MMR.
* **Ablations:** Show a small table: Pop, ItemKNN, ALS, FAISS‑content, **Hybrid LTR** (winner).

(If you use **MovieLens 25M**, you can also play with **Tag Genome** signals as features for explainability. The 25M page documents this extra file.) ([GroupLens][2])

---

## Starter snippets (drop‑in)

**Join MovieLens ↔ TMDB metadata**

```python
import pandas as pd

ml_ratings = pd.read_csv("ml-latest-small/ratings.csv")
ml_movies  = pd.read_csv("ml-latest-small/movies.csv")
ml_links   = pd.read_csv("ml-latest-small/links.csv")

tmdb_meta  = pd.read_csv("the-movies-dataset/movies_metadata.csv", low_memory=False)
tmdb_meta["id"] = pd.to_numeric(tmdb_meta["id"], errors="coerce").astype("Int64")

df = (ml_movies
      .merge(ml_links[["movieId","tmdbId"]], on="movieId", how="left")
      .merge(tmdb_meta[["id","overview","release_date","runtime","genres","original_language"]],
             left_on="tmdbId", right_on="id", how="left"))
```

*(Known: `id` parsing and some bad rows—coerce & drop. This is common with The Movies Dataset.)* ([Kaggle][4])

**Implicit ALS baseline**

```python
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k

# Build (items x users) matrix for implicit
user_map = {u:i for i,u in enumerate(ml_ratings.userId.unique())}
item_map = {m:i for i,m in enumerate(ml_ratings.movieId.unique())}

rows = ml_ratings.movieId.map(item_map).values
cols = ml_ratings.userId.map(user_map).values
data = (ml_ratings.rating >= 4).astype(float).values

X = sp.coo_matrix((data, (rows, cols))).tocsr()

model = AlternatingLeastSquares(factors=64, regularization=0.02, iterations=20)
model.fit(X)  # items x users
```

([Benfred][7])

**Content embeddings + FAISS**

```python
from transformers import AutoTokenizer, AutoModel
import torch, faiss, numpy as np

tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
mdl = AutoModel.from_pretrained("intfloat/e5-base-v2")  # or sentence-transformers/all-MiniLM-L6-v2

def embed(texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).bool()
        pooled = (out.masked_fill(~mask, 0).sum(1) / mask.sum(1))
    v = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    return v

item_vecs = embed(df["title"].fillna("") + " " + df["overview"].fillna("") + " " + df["genres"].fillna(""))
index = faiss.IndexFlatIP(item_vecs.shape[1])
index.add(item_vecs)  # inner product (vectors must be normalized)
```

([Hugging Face][8])

**LightGBM ranker skeleton**

```python
import lightgbm as lgb
# X_train: features; y_train: relevance (1 for held-out positive, 0 for sampled negatives)
# group_train: number of candidates per user in order
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63)
ranker.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val])
```

([lightgbm.readthedocs.io][10])

---

## Deliverables (interview‑ready)

* **/notebooks**: `01_data_join.ipynb`, `02_cf_baselines.ipynb`, `03_content_retrieval.ipynb`, `04_hybrid_ltr.ipynb`, `05_snack_pairing.ipynb`.
* **/service**: Minimal **FastAPI** with `/recommend` returning `[movie, reason, snack]`.
* **/report.md**: Tables for Recall\@10, NDCG\@10, ILS, ablation grid; a 1‑page **design note** on hybrid ranking and cold‑start.

**How you’ll talk about it:**

> “I built a hybrid recommender: ALS + content retrieval (FAISS) feeding a **LambdaMART** re‑ranker with cast/genre features—then paired each movie with a snack recipe using the same text embedding space. Offline, the **hybrid** beat ALS by +X NDCG\@10, and MMR improved diversity. We instrumented cold‑start by hiding popular titles and showed the content tower recovered Recall\@10.” ([Benfred][7])

---

## Common gotchas (so you don’t stall)

* **`movies_metadata.id` is stringy** and has a few malformed rows—`to_numeric(..., errors="coerce")` then drop NA. ([Kaggle][4])
* **`links.csv` mapping**: Not every MovieLens item has a TMDB ID; keep a left join and accept some misses. ([GroupLens Files][3])
* **Windows + implicit** sometimes needs build tools; if pip fails, use Conda or wheels (doc issues exist). ([GitHub][13])

---

## Nice extensions (pick any 1–2 if you have time later)

1. **Mood‑aware retrieval**: Encode the user mood prompt (“cozy rainy Sunday”) and add its cosine to the ranker. (Simple feature, big perceived quality.)
2. **MMR diversity** in the final 10: maximize `λ * relevance − (1−λ) * similarity`.
3. **Tag‑Genome features** (if you swap to ML‑25M): add a few calibrated tag relevances as ranker features; use them in explanations. ([GroupLens][2])
4. **Two‑tower DNN**: Instead of profile‑averaging, actually train a small two‑tower (user history text → vector, item text → vector) with in‑batch negatives, then keep LightGBM as the re‑ranker.
5. **Recipe ranker**: Learn a tiny LightGBM on Food.com interactions to pick snacks people rate highly for “quick” recipes; then blend with cosine similarity to the movie. ([Kaggle][12])

---

## If your goal shifts but you want similar skills

* **Double‑Feature Slate Builder**: recommend **pairs** of movies that are different yet complementary (optimize coverage + diversity with MMR).
* **“Trailer‑only” Recs**: Use TMDB videos or YouTube trailer captions to build a **video‑text** embedding tower; still re‑rank with metadata.
* **Book‑&‑Beverage Night**: Same pipeline, but Goodreads (ratings) + tea/coffee recipes (Food.com).

---

## Where to download quickly (recap)

* **MovieLens latest‑small / latest / 25M** (official GroupLens): stable links, CSVs. ([GroupLens][1])
* **The Movies Dataset** (Kaggle; TMDB‑derived metadata). ([Kaggle][4])
* **Food.com Recipes & Interactions** (Kaggle). ([Kaggle][5])

If you want, I can format this into a ready‑to‑run repo layout (folders + README + notebook stubs + a minimal API), but you can also follow the outline above and be fully productive right away.

[1]: https://grouplens.org/datasets/movielens/latest/?utm_source=chatgpt.com "MovieLens Latest Datasets"
[2]: https://grouplens.org/datasets/movielens/25m/?utm_source=chatgpt.com "MovieLens 25M Dataset"
[3]: https://files.grouplens.org/datasets/movielens/ml-20m-README.html?utm_source=chatgpt.com "ml-20m-README.html"
[4]: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?utm_source=chatgpt.com "The Movies Dataset"
[5]: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?utm_source=chatgpt.com "Food.com Recipes and Interactions"
[6]: https://www.ark.cs.cmu.edu/personas/?utm_source=chatgpt.com "CMU Movie Summary Corpus - Noah's ARK"
[7]: https://benfred.github.io/implicit/?utm_source=chatgpt.com "Implicit 0.6.1 documentation"
[8]: https://huggingface.co/intfloat/e5-base-v2?utm_source=chatgpt.com "intfloat/e5-base-v2"
[9]: https://github.com/facebookresearch/faiss?utm_source=chatgpt.com "facebookresearch/faiss: A library for efficient similarity ..."
[10]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html?utm_source=chatgpt.com "lightgbm.LGBMRanker — LightGBM 4.6.0.99 documentation"
[11]: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions/metadata?utm_source=chatgpt.com "Food.com Recipes and Interactions"
[12]: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data?select=RAW_recipes.csv&utm_source=chatgpt.com "Food.com Recipes and Interactions"
[13]: https://github.com/benfred/implicit/issues/361?utm_source=chatgpt.com "Failed to install implicit with pip · Issue #361"
