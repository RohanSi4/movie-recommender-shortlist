# Two-Tower Retrieval

The retrieval stage learns embeddings for users and movies in a shared 64-dim
space, so candidate generation becomes "find the items closest to this user"
instead of a hand-tuned heuristic. Ranking (LightGBM) then orders the
shortlist. This is the standard industrial retrieval-then-ranking shape.

## Architecture

~~~text
 user id ─► embedding ─┐                     ┌─ embedding ◄─ movie id
 genre prefs (train    ├─► MLP ─► 64d ─► L2  L2 ◄─ 64d ◄─ MLP ◄─┤ genre multi-hot
 window means),        │        normalize    normalize          │ release-era
 log activity,         ┘             \        /                 ┘ one-hot
 mean rating                      dot product / temperature
                                        │
                          in-batch sampled softmax loss
~~~

Each training example is one positive (user, movie) pair with rating >= 4.0.
Within a batch, every other item acts as a negative for that user.

## Split protocol

Identical to `build_training_dataset.py`: sort all ratings by timestamp and
hold out the most recent 10 percent as validation. Retrieval and ranking are
evaluated on the same held-out window, and user features are computed from the
training window only, so nothing from the future leaks into the model.

## Run it

~~~bash
# Train (auto-picks cuda > mps > cpu). Full data takes a GPU; smoke runs work anywhere.
python ml/scripts/train_two_tower.py \
  --processed-dir ml/data/processed \
  --out-dir ml/models/two_tower \
  --epochs 3

# Smoke run on a subsample
python ml/scripts/train_two_tower.py \
  --processed-dir ml/data/processed \
  --out-dir ml/models/two_tower_smoke \
  --sample-users 20000 --epochs 2

# Evaluate recall@k against popularity and random baselines
python ml/scripts/evaluate_retrieval.py \
  --processed-dir ml/data/processed \
  --model-dir ml/models/two_tower_smoke \
  --out ml/models/two_tower_smoke/retrieval_metrics.json
~~~

The evaluation excludes items a user already rated in the training window from
every method's candidates (standard protocol), reports recall@100 and
recall@500 over held-out positives, and includes catalog coverage because a
retriever that only ever surfaces popular titles scores deceptively well.

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
same exact retrieval scan. This gives a new visitor a useful personal query
without pretending they have an existing user embedding.
If `MODEL_API_BASE` is configured, LightGBM reranks the retrieved candidates;
if that service fails, the API returns the learned retrieval order.

Measured locally on an Apple M4 Pro over 200 requests after 20 warmups:

| path | client p50 | client p95 | server p50 |
|---|---:|---:|---:|
| known user | 4.3 ms | 5.8 ms | 3 ms |
| movie similarity | 3.5 ms | 3.7 ms | 3 ms |
| cold-user fallback | 0.4 ms | 0.4 ms | <1 ms |

## Honest limitations

- In-batch negatives sample negatives proportional to item popularity, which
  over-penalizes popular items. Training applies the standard log-Q
  sampled-softmax correction by default (subtract each item's log sampling
  probability from its logit column); `--no-logq-correction` reproduces the
  uncorrected ablation. The first full uncorrected run scored recall@100
  0.014 against a 0.127 popularity baseline, which is what motivated the
  correction.
- Users unseen in the training window have no embedding (cold start). The
  service keeps its popularity path for those.
- Smoke-run numbers are for wiring verification, not model quality claims.
  Full-data training numbers belong in the README only after a real run.
