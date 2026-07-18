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

The product-aligned test results are:

| supplied favorites | HitRate@10 | NDCG@10 | Recall@100 | popularity Recall@100 |
|---:|---:|---:|---:|---:|
| 1 | 82.1% | 0.361 | 0.282 | 0.250 |
| 3 | 84.7% | 0.358 | 0.319 | 0.237 |
| 5 | 84.1% | 0.338 | 0.331 | 0.228 |

The five-favorite model has 45 percent higher Recall@100 than popularity while
covering 14.5 percent of the catalog at depth 100, compared with 0.12 percent
for popularity. The raw reports also include precision, MRR, and every protocol
identifier needed to reproduce the cohort.

## Go serving path

`export_embeddings.py` publishes float16 user and item vectors plus a compact
history index containing every training-window rating for each warm user. The
manifest hashes all three files and identifies one model run. The Go service
verifies the full bundle at startup, widens vectors to float32, and performs an
exact dot-product scan over all 87,585 items. It uses a bounded top-k heap, so it
does not allocate and sort the full catalog on every request.

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

## Honest limitations

- The warm-user score applies only to the 35.2 percent of future-positive users
  who have a stored training embedding. The README leads with the anonymous
  taste test because that matches the public product.
- 42,538 catalog movies have no positive training interaction. Their content
  features help, but richer text features and explicit cold-item training remain
  important next steps.
- User-balanced in-batch negatives still include unlabeled positives from other
  histories. Explicit known-positive masking and hard negatives are the next
  modeling round.
- Users unseen in the stored-user vocabulary still take the popularity fallback
  unless they provide favorite movies.
- Smoke-run numbers are for wiring verification, not model quality claims.
  Full-data training numbers belong in the README only after a real run.
