#!/usr/bin/env python3
"""Evaluate two-tower retrieval vs popularity and random baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall@k.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing ratings.parquet.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing item_embeddings.parquet and user_embeddings.parquet.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of most-recent ratings held out (must match training).",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Minimum rating that counts as a positive interaction.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="100,500",
        help="Comma-separated k values for recall@k.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=2000,
        help="Cap on evaluated users (sampled with --seed) to bound runtime.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for user sampling and the random baseline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for metrics.",
    )
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def train_val_split(ratings: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Same global time-based split as build_training_dataset.py (the source of
    # truth for this protocol), so retrieval and ranking evals share a split.
    ratings_sorted = ratings.sort_values("timestamp")
    split_idx = int(len(ratings_sorted) * (1.0 - val_fraction))
    train = ratings_sorted.iloc[:split_idx]
    val = ratings_sorted.iloc[split_idx:]
    return train, val


def load_embeddings(path: Path, id_col: str) -> Tuple[List[int], np.ndarray]:
    frame = load_parquet(path)
    dims = [c for c in frame.columns if c.startswith("e")]
    return frame[id_col].tolist(), frame[dims].to_numpy(dtype=np.float32)


def recall_from_ranked(
    ranked_ids: List[int], truth: Set[int], ks: List[int]
) -> Dict[int, float]:
    hits = {k: 0 for k in ks}
    for rank, movie_id in enumerate(ranked_ids):
        if movie_id in truth:
            for k in ks:
                if rank < k:
                    hits[k] += 1
    return {k: hits[k] / len(truth) for k in ks}


def main() -> None:
    args = parse_args()
    ks = sorted(int(k) for k in args.ks.split(","))
    rng = np.random.default_rng(args.seed)

    ratings = load_parquet(args.processed_dir / "ratings.parquet")
    train_ratings, val_ratings = train_val_split(ratings, args.val_fraction)

    item_ids, item_matrix = load_embeddings(args.model_dir / "item_embeddings.parquet", "movieId")
    user_ids, user_matrix = load_embeddings(args.model_dir / "user_embeddings.parquet", "userId")
    user_row = {u: i for i, u in enumerate(user_ids)}
    item_id_arr = np.array(item_ids)

    val_positives = val_ratings[val_ratings["rating"] >= args.positive_threshold]
    truth_by_user = val_positives.groupby("userId")["movieId"].agg(set)

    # Only users the model has an embedding for (i.e. seen in the training
    # window) can be evaluated; cold-start users are a separate problem.
    eligible = [u for u in truth_by_user.index if u in user_row]
    if len(eligible) > args.max_users:
        eligible = list(rng.choice(eligible, size=args.max_users, replace=False))

    seen_by_user = (
        train_ratings[train_ratings["userId"].isin(eligible)]
        .groupby("userId")["movieId"]
        .agg(set)
    )

    # Popularity baseline: most-rated items in the training window.
    pop_counts = train_ratings["movieId"].value_counts()
    pop_ranked_all = [m for m in pop_counts.index.tolist() if m in set(item_ids)]

    per_method_recalls: Dict[str, Dict[int, List[float]]] = {
        m: {k: [] for k in ks} for m in ("two_tower", "popularity", "random")
    }
    coverage_sets: Dict[str, Set[int]] = {m: set() for m in per_method_recalls}
    max_k = max(ks)

    for user_id in eligible:
        truth = truth_by_user[user_id]
        # Standard protocol: exclude items the user already rated in the
        # training window from every method's candidate ranking.
        seen = seen_by_user.get(user_id, set())
        truth = truth - seen
        if not truth:
            continue

        scores = item_matrix @ user_matrix[user_row[user_id]]
        order = np.argsort(-scores)
        tt_ranked: List[int] = []
        for idx in order:
            movie_id = int(item_id_arr[idx])
            if movie_id in seen:
                continue
            tt_ranked.append(movie_id)
            if len(tt_ranked) >= max_k:
                break

        pop_ranked = [m for m in pop_ranked_all if m not in seen][:max_k]
        rand_ranked = [
            int(m)
            for m in rng.choice(item_id_arr, size=max_k + len(seen), replace=False)
            if m not in seen
        ][:max_k]

        for method, ranked in (
            ("two_tower", tt_ranked),
            ("popularity", pop_ranked),
            ("random", rand_ranked),
        ):
            recalls = recall_from_ranked(ranked, truth, ks)
            for k in ks:
                per_method_recalls[method][k].append(recalls[k])
            coverage_sets[method].update(ranked[: min(ks)])

    users_evaluated = len(per_method_recalls["two_tower"][ks[0]])
    metrics: Dict[str, object] = {
        "users_evaluated": users_evaluated,
        "catalog_size": len(item_ids),
    }
    for method in per_method_recalls:
        for k in ks:
            values = per_method_recalls[method][k]
            metrics[f"recall@{k}_{method}"] = float(np.mean(values)) if values else 0.0
        metrics[f"coverage@{min(ks)}_{method}"] = len(coverage_sets[method]) / len(item_ids)

    print(json.dumps(metrics, indent=2))

    header = "| method | " + " | ".join(f"recall@{k}" for k in ks) + f" | coverage@{min(ks)} |"
    rule = "|---" * (len(ks) + 2) + "|"
    print()
    print(header)
    print(rule)
    for method in ("two_tower", "popularity", "random"):
        cells = " | ".join(f"{metrics[f'recall@{k}_{method}']:.4f}" for k in ks)
        print(f"| {method} | {cells} | {metrics[f'coverage@{min(ks)}_{method}']:.4f} |")

    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
