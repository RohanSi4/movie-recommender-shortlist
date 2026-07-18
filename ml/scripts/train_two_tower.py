#!/usr/bin/env python3
"""Train a two-tower retrieval model on MovieLens ratings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# The canonical MovieLens genre list. "(no genres listed)" maps to all zeros.
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Release-era buckets derived from the year in the MovieLens title, e.g.
# "Toy Story (1995)". Index len(ERA_STARTS) is reserved for unknown years.
ERA_STARTS = [0, 1970, 1980, 1990, 2000, 2010, 2020]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing ratings.parquet and movies.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoint, embeddings, and metrics.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of most-recent ratings held out (must match build_training_dataset.py).",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Minimum rating that counts as a positive interaction.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Dimension of the shared embedding space.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Positive pairs per batch (also the in-batch negative count).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for in-batch sampled softmax.",
    )
    parser.add_argument(
        "--logq-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the log-Q sampled-softmax correction (disable for ablation).",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help="Optional cap on distinct training users (smoke runs).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional cap on training positives (smoke runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_val_split(ratings: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Same global time-based split as build_training_dataset.py (the source of
    # truth for this protocol), so retrieval and ranking evals share a split.
    ratings_sorted = ratings.sort_values("timestamp")
    split_idx = int(len(ratings_sorted) * (1.0 - val_fraction))
    train = ratings_sorted.iloc[:split_idx]
    val = ratings_sorted.iloc[split_idx:]
    return train, val


def extract_year(title: str) -> int:
    # MovieLens titles end with "(YYYY)"; return 0 when absent or malformed.
    if not isinstance(title, str) or len(title) < 6:
        return 0
    tail = title.strip()
    if tail.endswith(")") and tail[-6] == "(":
        year = tail[-5:-1]
        if year.isdigit():
            return int(year)
    return 0


def era_bucket(year: int) -> int:
    if year <= 0:
        return len(ERA_STARTS)
    bucket = 0
    for i, start in enumerate(ERA_STARTS):
        if year >= start:
            bucket = i
    return bucket


def build_item_features(movies: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Genre multi-hot and era one-hot per movie, aligned to movies row order."""
    genre_index = {g: i for i, g in enumerate(GENRES)}
    genre_mat = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    era_mat = np.zeros((len(movies), len(ERA_STARTS) + 1), dtype=np.float32)

    for row, (genres, title) in enumerate(zip(movies["genres"], movies["title"])):
        if isinstance(genres, str):
            for g in genres.split("|"):
                idx = genre_index.get(g)
                if idx is not None:
                    genre_mat[row, idx] = 1.0
        era_mat[row, era_bucket(extract_year(title))] = 1.0

    return genre_mat, era_mat


def build_user_features(
    train_positives: pd.DataFrame, genre_mat: np.ndarray, item_row: Dict[int, int]
) -> Tuple[np.ndarray, List[int]]:
    """Per-user genre preference means plus activity stats, train window only."""
    user_ids = sorted(train_positives["userId"].unique())
    user_row = {u: i for i, u in enumerate(user_ids)}
    n_genres = genre_mat.shape[1]

    genre_sum = np.zeros((len(user_ids), n_genres), dtype=np.float32)
    counts = np.zeros(len(user_ids), dtype=np.float32)
    rating_sum = np.zeros(len(user_ids), dtype=np.float32)

    u_idx = train_positives["userId"].map(user_row).to_numpy()
    m_idx = train_positives["movieId"].map(item_row).to_numpy()
    ratings = train_positives["rating"].to_numpy(dtype=np.float32)

    np.add.at(genre_sum, u_idx, genre_mat[m_idx])
    np.add.at(counts, u_idx, 1.0)
    np.add.at(rating_sum, u_idx, ratings)

    safe_counts = np.maximum(counts, 1.0)
    genre_pref = genre_sum / safe_counts[:, None]
    activity = np.log1p(counts)[:, None]
    mean_rating = (rating_sum / safe_counts)[:, None]

    features = np.concatenate([genre_pref, activity, mean_rating], axis=1)
    return features.astype(np.float32), user_ids


class Tower(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, embed_dim: int):
        super().__init__()
        self.id_embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.id_embedding(ids), features], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, user_feat_dim: int, n_items: int, item_feat_dim: int, embed_dim: int):
        super().__init__()
        self.user_tower = Tower(n_users, user_feat_dim, embed_dim)
        self.item_tower = Tower(n_items, item_feat_dim, embed_dim)


def in_batch_softmax_loss(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    # Each row's positive item is scored against every other item in the batch
    # as an implicit negative. In-batch sampling over-penalizes popular items
    # (they appear as negatives in proportion to their frequency), so when
    # log_q is provided we apply the sampled-softmax correction: subtract each
    # item's log sampling probability from its logit column. See
    # docs/RETRIEVAL.md.
    logits = user_vecs @ item_vecs.T / temperature
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)


def export_embeddings(
    out_path: Path, id_col: str, ids: List[int], vectors: np.ndarray
) -> None:
    frame = pd.DataFrame({id_col: ids})
    for d in range(vectors.shape[1]):
        frame[f"e{d}"] = vectors[:, d]
    frame.to_parquet(out_path, index=False)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    started = time.time()

    ratings = load_parquet(args.processed_dir / "ratings.parquet")
    movies = load_parquet(args.processed_dir / "movies.parquet")

    train_ratings, _ = train_val_split(ratings, args.val_fraction)
    positives = train_ratings[train_ratings["rating"] >= args.positive_threshold]

    if args.sample_users:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(
            positives["userId"].unique(),
            size=min(args.sample_users, positives["userId"].nunique()),
            replace=False,
        )
        positives = positives[positives["userId"].isin(keep)]
    if args.sample_rows:
        positives = positives.sample(
            n=min(args.sample_rows, len(positives)), random_state=args.seed
        )

    item_ids: List[int] = movies["movieId"].tolist()
    item_row = {m: i for i, m in enumerate(item_ids)}
    positives = positives[positives["movieId"].isin(item_row)]

    genre_mat, era_mat = build_item_features(movies)
    item_features = np.concatenate([genre_mat, era_mat], axis=1)
    user_features, user_ids = build_user_features(positives, genre_mat, item_row)
    user_row = {u: i for i, u in enumerate(user_ids)}

    print(f"Training on {len(positives)} positives, {len(user_ids)} users, {len(item_ids)} items ({device}).")

    model = TwoTower(
        n_users=len(user_ids),
        user_feat_dim=user_features.shape[1],
        n_items=len(item_ids),
        item_feat_dim=item_features.shape[1],
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    u_idx = torch.tensor(positives["userId"].map(user_row).to_numpy(), dtype=torch.long)
    m_idx = torch.tensor(positives["movieId"].map(item_row).to_numpy(), dtype=torch.long)
    user_feats_t = torch.tensor(user_features)
    item_feats_t = torch.tensor(item_features)

    # Empirical sampling probability of each item among training positives,
    # used for the log-Q sampled-softmax correction (clamped so unseen items
    # in the catalog never produce -inf).
    item_counts = np.bincount(m_idx.numpy(), minlength=len(item_ids)).astype(np.float64)
    item_q = np.clip(item_counts / max(1, len(m_idx)), 1e-9, None)
    log_q_all = torch.tensor(np.log(item_q), dtype=torch.float32)

    epoch_losses: List[float] = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(u_idx))
        losses: List[float] = []
        for start in range(0, len(perm), args.batch_size):
            batch = perm[start : start + args.batch_size]
            if len(batch) < 2:
                continue
            bu, bm = u_idx[batch], m_idx[batch]
            user_vecs = model.user_tower(bu.to(device), user_feats_t[bu].to(device))
            item_vecs = model.item_tower(bm.to(device), item_feats_t[bm].to(device))
            batch_log_q = log_q_all[bm].to(device) if args.logq_correction else None
            loss = in_batch_softmax_loss(
                user_vecs, item_vecs, args.temperature, log_q=batch_log_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
        print(f"  epoch {epoch + 1}/{args.epochs} loss {epoch_losses[-1]:.4f}")

    ensure_out_dir(args.out_dir)

    model.eval()
    with torch.no_grad():
        item_vecs_all: List[np.ndarray] = []
        for start in range(0, len(item_ids), 8192):
            ids = torch.arange(start, min(start + 8192, len(item_ids)), dtype=torch.long)
            vecs = model.item_tower(ids.to(device), item_feats_t[ids].to(device))
            item_vecs_all.append(vecs.cpu().numpy())
        item_matrix = np.concatenate(item_vecs_all).astype(np.float32)

        user_vecs_all: List[np.ndarray] = []
        for start in range(0, len(user_ids), 8192):
            ids = torch.arange(start, min(start + 8192, len(user_ids)), dtype=torch.long)
            vecs = model.user_tower(ids.to(device), user_feats_t[ids].to(device))
            user_vecs_all.append(vecs.cpu().numpy())
        user_matrix = np.concatenate(user_vecs_all).astype(np.float32)

    checkpoint_path = args.out_dir / "two_tower.pt"
    items_path = args.out_dir / "item_embeddings.parquet"
    users_path = args.out_dir / "user_embeddings.parquet"
    metrics_path = args.out_dir / "training_metrics.json"

    torch.save({"state_dict": model.state_dict(), "config": vars(args) | {"out_dir": str(args.out_dir), "processed_dir": str(args.processed_dir)}}, checkpoint_path)
    export_embeddings(items_path, "movieId", item_ids, item_matrix)
    export_embeddings(users_path, "userId", user_ids, user_matrix)

    metrics = {
        "device": str(device),
        "runtime_seconds": round(time.time() - started, 1),
        "epoch_losses": epoch_losses,
        "train_positives": int(len(positives)),
        "users": len(user_ids),
        "items": len(item_ids),
        "embed_dim": args.embed_dim,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "positive_threshold": args.positive_threshold,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Training complete.")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Item embeddings: {items_path}")
    print(f"  User embeddings: {users_path}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
