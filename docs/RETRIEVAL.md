# Two-Tower Retrieval

The retrieval stage learns embeddings for users and movies in a shared 64-dim
space. It trains both the stored-user path and the public product path, where a
visitor picks one to five favorite movies and asks for a personal shortlist.

## Architecture

~~~text
user id + train-window features ─► user tower ─► 64d ─┐
                                                       ├─► sampled softmax
1-to-5 liked movies ─► mean + normalize ─► taste 64d ─┘          ▲
                                                                  │
movie id + genre + era ───────────────► shared item tower ─► 64d ─┘
~~~

Each target is a movie rated at least 4.0. Batches sample users uniformly, then
sample one target per user, so highly active viewers do not dominate a metric
that averages across viewers. Target movies are de-duplicated before scoring.
Log-Q uses the actual user-balanced target probability instead of the raw
interaction frequency.

The second objective samples one to five other liked movies from the same
training history, averages their current item vectors exactly like the Go API,
and predicts the target. That makes the live taste builder a trained query, not
an incidental use of item embeddings.

## Split protocol

Training sorts all ratings by timestamp and freezes the earliest 90 percent at
the 2020-11-05 cutoff. User features are computed from that window only.

Two evaluations answer different questions:

- Warm-user retrieval measures stored MovieLens user embeddings against future
  positives, with training history removed from every method.
- Taste retrieval finds users with no pre-cutoff history and at least six future
  positive ratings. Their earliest one, three, or five favorites become the
  supplied seeds and their later favorites become truth. A deterministic split
  creates 7,154 validation users and an untouched 7,060-user test cohort.

## Run it

~~~bash
# Train (auto-picks cuda > mps > cpu). Full data takes a GPU; smoke runs work anywhere.
python ml/scripts/train_two_tower.py \
  --processed-dir ml/data/processed \
  --out-dir ml/models/two_tower_taste \
  --epochs 3 --batch-size 1024 \
  --sampling-strategy user-balanced \
  --taste-loss-weight 0.5

# Smoke run on a subsample
python ml/scripts/train_two_tower.py \
  --processed-dir ml/data/processed \
  --out-dir ml/models/two_tower_taste_smoke \
  --sample-users 20000 --epochs 2 --batch-size 1024

# Evaluate stored users
python ml/scripts/evaluate_retrieval.py \
  --processed-dir ml/data/processed \
  --model-dir ml/models/two_tower_taste \
  --out docs/metrics/retrieval_eval.json

# Evaluate the untouched anonymous test cohort
python ml/scripts/evaluate_taste_retrieval.py \
  --processed-dir ml/data/processed \
  --model-dir ml/models/two_tower_taste \
  --cohort test --max-users 0 \
  --out docs/metrics/taste_eval_test.json
~~~

### Provenance of the published model

`make train-retrieval` trains three epochs from scratch and is the recipe to use
for a fresh run. The specific checkpoint behind the numbers below was built
incrementally instead, so the command above will land near these results rather
than exactly on them:

1. `two_tower_logq`, six epochs at batch 4096 with the warm-user objective only.
2. `two_tower_taste_full_e1`, one epoch of taste fine-tuning at batch 1024 with
   `--taste-loss-weight 0.5`, initialized from that checkpoint with seed 42.
3. `two_tower_taste_full_e2`, one more epoch initialized from e1 with seed 43.
   `ml/models/two_tower_taste` is that checkpoint, and its
   `item_embeddings.parquet` hash is the one stamped into every metrics file.

The serving model is simpler: `make train-serving` reproduces it directly.

The product-aligned test results are:

| supplied favorites | HitRate@10 | popularity | NDCG@10 | Recall@100 | popularity | coverage@100 | popularity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.1% | 81.9% | 0.361 | 0.282 | 0.250 | 27.0% | 0.12% |
| 3 | 84.7% | 77.2% | 0.358 | 0.319 | 0.237 | 19.1% | 0.12% |
| 5 | 84.1% | 73.8% | 0.338 | 0.331 | 0.228 | 14.5% | 0.12% |

The five-favorite model has 45 percent higher Recall@100 than popularity while
covering 14.5 percent of the catalog at depth 100, compared with 0.12 percent
for popularity. The raw reports also include precision, MRR, and every protocol
identifier needed to reproduce the cohort.

Three honest readings of that table:

- **HitRate@10 is a generous metric on this cohort.** Eligible users have at
  least six future positives and in practice far more: the median truth set is
  about 50 movies. Landing one of 50 in a top 10 is not a high bar, which is
  why popularity alone clears it 73.8 percent of the time. Recall@100 and
  catalog coverage are the metrics that actually separate the methods.
- **At one seed the model and popularity are effectively tied on HitRate@10**,
  82.1 against 81.9. A single movie is a thin query. The model still wins
  Recall@100 at that seed count, and it answers from 27 percent of the catalog
  against popularity's 0.12 percent, so it is personalizing rather than showing
  everyone the same canon. The gap on HitRate@10 opens only once there is a mix
  to blend, which is the case the product is built around.
- **HitRate@10 peaks at three seeds, not five.** Averaging more seed vectors
  moves the query toward the center of a viewer's taste. That keeps helping deep
  recall (0.282, 0.319, 0.331) and costs a little top-10 sharpness.

Model selection used the validation cohort only. Five candidate runs were scored
there (`ml/models/*/taste_eval_validation.json`); the selected run was scored on
the 7,060-user test cohort once. Validation HitRate@10 at five seeds was 0.8365
and test was 0.8407, so the held-out number did not come from tuning against it.

## Go serving path

`export_embeddings.py` publishes float16 user and item vectors plus a compact
history index containing every training-window rating for each warm user. The
manifest hashes all three files and identifies one model run. The Go service
verifies the full bundle at startup, widens vectors to float32, and performs an
exact dot-product scan over all 89,585 items (the 87,585-movie MovieLens catalog
plus 2,000 recent releases discovered from TMDB). It uses a bounded top-k heap,
so it does not allocate and sort the full catalog on every request.

Known-user requests exclude the user's stored history before returning the
shortlist. Unknown users take the existing popularity fallback. Movie-based
queries use item-to-item cosine similarity and always exclude the seed title.
The public taste builder accepts one to five movies, averages their item
vectors, normalizes the combined query, and excludes every seed before the
same exact retrieval scan. This is the same query construction used during
training and evaluation.
If `MODEL_API_BASE` is configured, LightGBM reranks the retrieved candidates;
if that service fails, the API returns the learned retrieval order.

Measured locally on an Apple M4 Pro over 200 requests after 20 warmups:

| path | client p50 | client p95 | server p50 |
|---|---:|---:|---:|
| known user | 4.0 ms | 5.6 ms | 3 ms |
| movie similarity | 3.4 ms | 3.9 ms | 3 ms |
| cold-user fallback | 0.4 ms | 0.4 ms | <1 ms |

## Cold seeds and the popularity blend

The offline split only tests warm seeds: every evaluated favorite predates the
2020-11-05 cutoff and therefore has a trained embedding. Real visitors do the
opposite and seed the shortlist with the recent blockbuster they just watched. A
movie released after the cutoff has no training positives, so its embedding is
untrained noise and its nearest neighbors are effectively random obscure titles.
That, not a missing popularity prior, was the main cause of weak recommendations
from recent seeds.

Two changes address it, and they are deliberately separate.

**Evaluate on a holdout, deploy on all the data.** The published metrics come
from the temporal-holdout model (`val_fraction 0.1`) and never move. The bundle
the Go service loads is a second model trained on nearly the whole timeline
(`make train-serving`, `val_fraction 0.01`) so 2021-2023 releases get real
embeddings. Everything but the tiny final holdout becomes training signal.

**Cold-gated popularity.** `export_embeddings.py` also writes `item_stats.bin`,
the count of training-window positive ratings (>= 4.0) per movie. That single
file feeds two derived signals in Go:

- a popularity score, the z-scored `log1p(count)`, and
- a per-seed warmth, `log1p(support) / log1p(WARM_REF)` clamped to `[0, 1]`,
  with `WARM_REF` defaulting to 300 positives.

The serving score is `dot + weight * popularity`, where the blend weight is
`COLD_POP_WEIGHT * (1 - mean seed warmth)` (default `COLD_POP_WEIGHT` 0.6). The
mean is taken because the taste query is an equal-weight average of the seeds, so
one warm seed should not mask a noisy one. Fully warm seeds drive the weight to
zero and keep pure personalization; fully cold seeds get the full popularity
rescue. Both the taste path and the movie-similarity path apply it. The blend is
backward compatible: with no stats file the weight is zero and the score is the
old exact dot product.

Leaning on popularity is a serving-only choice with a measured cost. A global
popularity weight monotonically hurts the honest held-out metric, dropping
validation-cohort HitRate@10 from 0.838 at weight 0 to 0.799 at 0.5. Gating on
warmth confines that cost to seeds the model genuinely cannot represent.

Observed behavior on the serving bundle for a visitor seeding Oppenheimer (2023,
zero training support) and Everything Everywhere All at Once (2022, now warm):
the pair returns Arrival, Parasite, Dune, and Her; Oppenheimer alone falls back
to the popular canon of Shawshank, Pulp Fiction, and The Matrix; a warm control
of Inception and The Matrix is untouched, returning The Dark Knight, Fight Club,
and the Lord of the Rings trilogy.

## Recent releases past the ratings wall

MovieLens stops in 2023, so no 2024+ film is even in the catalog: a visitor
cannot search for it, let alone seed with it. `discover_tmdb.py` closes that gap
by pulling recent releases from TMDB's `/discover/movie` endpoint (filtered by
release date and a minimum vote count) and writing them in the catalog's own
shape, keyed by a synthetic `movieId` of `100000000 + tmdbId` so the two id
spaces never collide.

A discovered movie has no interactions, so it cannot have a learned embedding.
The export gives it a **genre-centroid cold embedding** instead: for each
MovieLens genre, the mean of the trained item vectors carrying that genre, then a
new movie is placed at the normalized average of the centroids for its own
genres. It is the poor-man's content tower, and it is enough to make the film a
coherent seed and a retrievable candidate. Because its training support is zero,
the cold-seed blend automatically leans on popularity for it.

The effect, measured on a scratch bundle seeded with real 2024 titles: Dune: Part
Two returns Iron Man, Empire Strikes Back, and Guardians of the Galaxy; Inside
Out 2 returns WALL-E, Monsters Inc., and Up; Terrifier 3 returns Alien, The
Shining, and The Thing. Search also falls back to TMDB popularity for these
zero-rating titles so they are findable, not buried beneath obscure older matches.

The flow is `make discover-tmdb` (needs `TMDB_API_KEY`) followed by the normal
`make export` and `make export-retrieval`; both pick up the discovered CSV
automatically and are no-ops without it.

## Honest limitations

- The warm-user score applies only to the 35.2 percent of future-positive users
  who have a stored training embedding. The README leads with the anonymous
  taste test because that matches the public product.
- On the serving split, 35,591 of the 89,585 movies in the served bundle still
  have no positive training interaction (42,538 of 87,585 on the evaluation
  split). The cold-gated blend gives those seeds a sensible popular fallback,
  but richer text features and explicit cold-item training remain important
  next steps.
- The data itself ends in October 2023, so the newest releases stay thin no
  matter the split. Oppenheimer (July 2023) has zero training support even in the
  serving model and is served entirely by the popularity blend; a fresher ratings
  dump is the only real fix.
- User-balanced in-batch negatives still include unlabeled positives from other
  histories. Explicit known-positive masking and hard negatives are the next
  modeling round.
- Users unseen in the stored-user vocabulary still take the popularity fallback
  unless they provide favorite movies.
- Smoke-run numbers are for wiring verification, not model quality claims.
  Full-data training numbers belong in the README only after a real run.
