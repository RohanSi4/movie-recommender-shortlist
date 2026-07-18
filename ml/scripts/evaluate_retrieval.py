#!/usr/bin/env python3
"""Evaluate two-tower retrieval vs popularity and random baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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
        default="10,50,100,500",
        help="Comma-separated k values for recall@k.",
    )
    parser.add_argument(
        "--coverage-k",
        type=int,
        default=100,
        help="Depth used for catalog coverage.",
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


def top_k_metrics(ranked_ids: List[int], truth: Set[int]) -> Dict[str, float]:
    hits = np.array([movie_id in truth for movie_id in ranked_ids[:10]], dtype=np.float64)
    hit_ranks = np.flatnonzero(hits)
    ideal_length = min(10, len(truth))
    ideal_dcg = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
    return {
        "hit_rate@10": float(hits.any()),
        "precision@10": float(hits.sum() / 10),
        "ndcg@10": float((hits / np.log2(np.arange(2, 12))).sum() / ideal_dcg),
        "mrr@10": float(1.0 / (hit_ranks[0] + 1)) if len(hit_ranks) else 0.0,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    ks = sorted(int(k) for k in args.ks.split(","))
    rng = np.random.default_rng(args.seed)

    ratings = load_parquet(args.processed_dir / "ratings.parquet")
    train_ratings, val_ratings = train_val_split(ratings, args.val_fraction)
    cutoff = int(val_ratings.sort_values("timestamp", kind="stable").iloc[0]["timestamp"])

    item_ids, item_matrix = load_embeddings(args.model_dir / "item_embeddings.parquet", "movieId")
    user_ids, user_matrix = load_embeddings(args.model_dir / "user_embeddings.parquet", "userId")
    user_row = {u: i for i, u in enumerate(user_ids)}
    item_id_arr = np.array(item_ids)

    val_positives = val_ratings[val_ratings["rating"] >= args.positive_threshold]
    truth_by_user = val_positives.groupby("userId")["movieId"].agg(set)

    # Only users the model has an embedding for (i.e. seen in the training
    # window) can be evaluated; cold-start users are a separate problem.
    eligible = [u for u in truth_by_user.index if u in user_row]
    eligible_all = list(eligible)
    if args.max_users > 0 and len(eligible) > args.max_users:
        eligible = list(rng.choice(eligible, size=args.max_users, replace=False))

    seen_by_user = (
        train_ratings[train_ratings["userId"].isin(eligible)]
        .groupby("userId")["movieId"]
        .agg(set)
    )

    # Popularity baseline over the same explicit item universe. Items without
    # training ratings sort last instead of silently disappearing.
    pop_counts = train_ratings["movieId"].value_counts()
    pop_values = np.array([pop_counts.get(movie_id, 0) for movie_id in item_ids])
    pop_order = np.lexsort((item_id_arr, -pop_values))
    pop_ranked_all = [int(item_id_arr[index]) for index in pop_order]

    per_method_recalls: Dict[str, Dict[int, List[float]]] = {
        m: {k: [] for k in ks} for m in ("two_tower", "popularity", "random")
    }
    coverage_sets: Dict[str, Set[int]] = {m: set() for m in per_method_recalls}
    top10_metrics: Dict[str, Dict[str, List[float]]] = {
        method: {metric: [] for metric in ("hit_rate@10", "precision@10", "ndcg@10", "mrr@10")}
        for method in per_method_recalls
    }
    max_k = max(max(ks), args.coverage_k, 10)
    item_row = {movie_id: row for row, movie_id in enumerate(item_ids)}

    for user_id in eligible:
        truth = truth_by_user[user_id]
        # Standard protocol: exclude items the user already rated in the
        # training window from every method's candidate ranking.
        seen = seen_by_user.get(user_id, set())
        truth = truth - seen
        if not truth:
            continue

        scores = item_matrix @ user_matrix[user_row[user_id]]
        seen_rows = [item_row[movie_id] for movie_id in seen if movie_id in item_row]
        if seen_rows:
            scores[seen_rows] = -np.inf
        top_rows = np.argpartition(scores, -max_k)[-max_k:]
        top_rows = top_rows[np.argsort(-scores[top_rows], kind="stable")]
        tt_ranked = [int(item_id_arr[index]) for index in top_rows]

        pop_ranked = [m for m in pop_ranked_all if m not in seen][:max_k]
        rand_ranked = [int(m) for m in rng.permutation(item_id_arr) if m not in seen][:max_k]

        for method, ranked in (
            ("two_tower", tt_ranked),
            ("popularity", pop_ranked),
            ("random", rand_ranked),
        ):
            recalls = recall_from_ranked(ranked, truth, ks)
            for k in ks:
                per_method_recalls[method][k].append(recalls[k])
            for metric, value in top_k_metrics(ranked, truth).items():
                top10_metrics[method][metric].append(value)
            coverage_sets[method].update(ranked[: args.coverage_k])

    users_evaluated = len(per_method_recalls["two_tower"][ks[0]])
    metrics: Dict[str, object] = {
        "users_evaluated": users_evaluated,
        "validation_positive_users": len(truth_by_user),
        "eligible_warm_users": len(eligible_all),
        "warm_user_coverage": len(eligible_all) / max(1, len(truth_by_user)),
        "validation_positive_rows": len(val_positives),
        "eligible_validation_positive_rows": int(val_positives["userId"].isin(eligible_all).sum()),
        "catalog_size": len(item_ids),
        "cutoff_timestamp": cutoff,
        "cutoff_utc": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
        "val_fraction": args.val_fraction,
        "positive_threshold": args.positive_threshold,
        "sample_seed": args.seed,
        "sampled_user_sha256": hashlib.sha256(",".join(map(str, eligible)).encode()).hexdigest(),
        "item_embeddings_sha256": file_sha256(args.model_dir / "item_embeddings.parquet"),
    }
    metrics["eligible_positive_row_coverage"] = (
        metrics["eligible_validation_positive_rows"] / max(1, metrics["validation_positive_rows"])
    )
    for method in per_method_recalls:
        for k in ks:
            values = per_method_recalls[method][k]
            metrics[f"recall@{k}_{method}"] = float(np.mean(values)) if values else 0.0
        for metric, values in top10_metrics[method].items():
            metrics[f"{metric}_{method}"] = float(np.mean(values)) if values else 0.0
        metrics[f"coverage@{args.coverage_k}_{method}"] = len(coverage_sets[method]) / len(item_ids)

    print(json.dumps(metrics, indent=2))

    header = "| method | " + " | ".join(f"recall@{k}" for k in ks) + f" | hit@10 | ndcg@10 | coverage@{args.coverage_k} |"
    rule = "|---" * (len(ks) + 4) + "|"
    print()
    print(header)
    print(rule)
    for method in ("two_tower", "popularity", "random"):
        cells = " | ".join(f"{metrics[f'recall@{k}_{method}']:.4f}" for k in ks)
        print(
            f"| {method} | {cells} | {metrics[f'hit_rate@10_{method}']:.4f} | "
            f"{metrics[f'ndcg@10_{method}']:.4f} | {metrics[f'coverage@{args.coverage_k}_{method}']:.4f} |"
        )

    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
