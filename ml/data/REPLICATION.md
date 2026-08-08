# Regenerating the deleted training artifacts

`ml/data/processed/training/` was deleted on **2026-08-07** to reclaim 5.5 GB. It
held pure derived data, and everything needed to rebuild it byte-for-byte is
either still in this repo or is a public download. This file is the recipe.

Nothing that is expensive or impossible to reproduce was removed. See
"What was deliberately kept" at the bottom for why that distinction matters.

---

## What was deleted

| File | Size | Rows | Columns |
|---|---|---|---|
| `ml/data/processed/training/train.parquet` | 4.8 GB | 28,800,183 | 24 |
| `ml/data/processed/training/val.parquet` | 690 MB | 3,200,021 | 24 |

The two row counts sum to **32,000,204**, which is exactly the MovieLens ml-32m
rating count. Every rating becomes one row; the split is a 90/10 temporal cut,
not a random one (`build_training_dataset.py:train_val_split` sorts by
`timestamp`, then takes `iloc[:int(n * 0.9)]` as train and the remainder as val).
A random split would leak future ratings into training and inflate every metric,
so if a rebuild ever disagrees with the published numbers, check this first.

Schema (24 columns, identical in both files):

```
userId, movieId, label, timestamp,
rating_count_user, rating_mean_user, rating_std_user, last_rating_ts_user,
title, genres,
rating_count_movie, rating_mean_movie, rating_std_movie, last_rating_ts_movie,
tmdbId, tmdb_found, tmdb_release_date, tmdb_runtime, tmdb_vote_avg,
tmdb_vote_count, tmdb_popularity, tmdb_genres, tmdb_poster_path, tmdb_overview
```

---

## Rebuild

**One command**, because its inputs (`ml/data/processed/*.parquet` and
`ml/data/processed/features/`) were all kept:

```bash
make training
```

which expands to:

```bash
python ml/scripts/build_training_dataset.py \
  --processed-dir ml/data/processed \
  --features-dir  ml/data/processed/features \
  --out-dir       ml/data/processed/training
```

`--val-fraction` defaults to `0.1`. Do not pass a different value unless you
intend to invalidate comparisons against the published metrics.

### If `ml/data/processed/` is ever lost too

Then the chain restarts from the raw dataset:

```bash
make ingest    # ml/data/raw CSVs -> ml/data/processed/*.parquet
make enrich    # TMDB enrichment, needs TMDB_API_KEY  (SLOW, rate limited)
make features  # -> ml/data/processed/features/
make training  # -> ml/data/processed/training/
```

**Gotcha:** `make ingest` reads `--raw-dir ml/data/raw`, but that directory is
empty. The MovieLens CSVs actually live in `data/ml-32m/`. Either point the flag
at `data/ml-32m` or copy the four CSVs into `ml/data/raw` first.

---

## Source dataset

**MovieLens 32M (`ml-32m`)**, generated 2023-10-13 by GroupLens, University of
Minnesota. 32,000,204 ratings and 2,000,072 tag applications across 87,585
movies from 200,948 users, covering 1995-01-09 to 2023-10-12. Every included
user rated at least 20 movies. No demographic data.

Download: <https://files.grouplens.org/datasets/movielens/ml-32m.zip>
(landing page: <https://grouplens.org/datasets/movielens/>)

**`data/ml-32m/` is no longer on disk.** Both the `ml-32m.zip` archive and the
extracted CSVs were deleted on 2026-08-07 (1.1 GB combined). Re-download from the
URL above and unzip to `data/ml-32m/` if the chain ever has to restart from raw.

Deleting them is only safe because `ml/data/processed/*.parquet`, which is what
`ingest` produces from these CSVs, was kept. Nothing in the normal rebuild path
touches the raw files.

MD5 checksums, transcribed here from the dataset's own `checksums.txt` **before
that file was deleted along with the rest of the directory** — this table is now
the only copy in the repo:

```
8f033867bcb4e6be8792b21468b4fa6e  links.csv
0df90835c19151f9d819d0822e190797  movies.csv
cf12b74f9ad4b94a011f079e26d4270a  ratings.csv
963bf4fa4de6b8901868fddd3eb54567  tags.csv
```

Verify a fresh download with `md5 -r data/ml-32m/*.csv` (macOS) or
`md5sum` (Linux) before trusting a rebuild. A mismatch means GroupLens has
published a different revision, and every published metric in `docs/` was
computed against the checksums above.

---

## What was deliberately kept, and why

These are NOT cheaply reproducible, so they stayed:

- **`ml/data/processed/tmdb_enriched.csv`** (37 MB) and
  **`tmdb_discovered.csv`** need `TMDB_API_KEY` plus a long, rate-limited crawl.
  Losing these costs hours and an API key, not one `make` invocation.
- **`ml/data/processed/features/`** (29 MB) is the direct input to
  `make training`, and keeping it is what reduces the rebuild to one command.
- **`ml/data/processed/*.parquet`** (229 MB) is the ingested MovieLens core.
- **`ml/models/two_tower_taste_serving/`** and **`two_tower_taste/`**, the only
  two checkpoints any Makefile target loads.
- **`service/data/*.bin`** are tracked in git and are what the Go service
  actually serves. They do not depend on the training parquets at runtime.

Six other checkpoints (`two_tower_smoke`, `two_tower_taste_smoke`,
`two_tower_taste_full_e1`, `two_tower_taste_full_e2`, `two_tower_full`,
`two_tower_logq`) were deleted the same day. They were smoke runs, superseded
intermediate epochs, and abandoned variants, referenced nowhere except as prose
in `docs/RETRIEVAL.md`, which still carries the commands that produced them.
