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
