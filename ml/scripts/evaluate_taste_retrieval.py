#!/usr/bin/env python3
"""Evaluate the anonymous 1-to-5 favorite-movie flow on future users."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate favorite-movie taste retrieval.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--seed-counts", default="1,3,5")
    parser.add_argument("--cohort", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--popularity-weight", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def embedding_columns(frame: pd.DataFrame) -> List[str]:
    dims = [(int(column[1:]), column) for column in frame if column[1:].isdigit() and column.startswith("e")]
    dims.sort()
    if [index for index, _ in dims] != list(range(len(dims))):
        raise ValueError("Embedding columns must be contiguous e0..eN")
    return [column for _, column in dims]


def global_time_split(
    ratings: pd.DataFrame, val_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    ordered = ratings.sort_values("timestamp", kind="stable")
    split_index = int(len(ordered) * (1.0 - val_fraction))
    cutoff = int(ordered.iloc[split_index]["timestamp"])
    return ordered.iloc[:split_index], ordered.iloc[split_index:], cutoff


def cohort_bucket(user_id: int, split_seed: int) -> str:
    digest = hashlib.sha256(f"{split_seed}:{user_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "little") % 2 == 0 else "test"


def select_cold_users(
    train: pd.DataFrame,
    future_positives: pd.DataFrame,
    min_positives: int,
    cohort: str,
    split_seed: int,
    max_users: int,
) -> Tuple[List[int], Dict[int, List[int]], Dict[str, int]]:
    train_users = set(train["userId"].unique())
    ordered = future_positives.sort_values(["userId", "timestamp"], kind="stable")
    positives_by_user = ordered.groupby("userId")["movieId"].agg(list)
    eligible_all = sorted(
        int(user_id)
        for user_id, movies in positives_by_user.items()
        if user_id not in train_users and len(movies) >= min_positives
    )
    validation = [u for u in eligible_all if cohort_bucket(u, split_seed) == "validation"]
    test = [u for u in eligible_all if cohort_bucket(u, split_seed) == "test"]
    selected = eligible_all if cohort == "all" else (validation if cohort == "validation" else test)
    if max_users > 0:
        selected = selected[:max_users]
    histories = {user_id: [int(movie) for movie in positives_by_user[user_id]] for user_id in selected}
    return selected, histories, {
        "eligible_all": len(eligible_all),
        "eligible_validation": len(validation),
        "eligible_test": len(test),
    }


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    candidates = np.argpartition(scores, -k)[-k:]
    return candidates[np.argsort(-scores[candidates], kind="stable")]


def ranking_metrics(ranked: Sequence[int], truth: Set[int]) -> Dict[str, float]:
    hits = np.array([movie_id in truth for movie_id in ranked], dtype=np.float64)
    top10 = hits[:10]
    ideal_length = min(10, len(truth))
    ideal_dcg = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
    hit_ranks = np.flatnonzero(top10)
    return {
        "hit_rate@10": float(top10.any()),
        "precision@10": float(top10.mean()),
        "recall@10": float(top10.sum() / len(truth)),
        "recall@50": float(hits[:50].sum() / len(truth)),
        "recall@100": float(hits[:100].sum() / len(truth)),
        "ndcg@10": float((top10 / np.log2(np.arange(2, 12))).sum() / ideal_dcg),
        "mrr@10": float(1.0 / (hit_ranks[0] + 1)) if len(hit_ranks) else 0.0,
    }


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    seed_counts = sorted({int(value) for value in args.seed_counts.split(",")})
    if not seed_counts or seed_counts[0] < 1:
        raise ValueError("seed-counts must contain positive integers")
    if args.popularity_weight < 0:
        raise ValueError("popularity-weight must be non-negative")

    ratings = load_parquet(args.processed_dir / "ratings.parquet")
    train, future, cutoff = global_time_split(ratings, args.val_fraction)
    future_positives = future[future["rating"] >= args.positive_threshold]

    item_path = args.model_dir / "item_embeddings.parquet"
    item_frame = load_parquet(item_path)
    dims = embedding_columns(item_frame)
    item_ids = item_frame["movieId"].to_numpy(dtype=np.int64)
    item_matrix = item_frame[dims].to_numpy(dtype=np.float32)
    item_rows = {int(movie_id): row for row, movie_id in enumerate(item_ids)}

    users, histories, cohort_counts = select_cold_users(
        train,
        future_positives,
        max(seed_counts) + 1,
        args.cohort,
        args.split_seed,
        args.max_users,
    )
    if not users:
        raise ValueError("No cold users qualify for the requested cohort")

    popularity = train["movieId"].value_counts()
    popularity_scores = np.log1p(
        np.array([popularity.get(movie_id, 0) for movie_id in item_ids], dtype=np.float32)
    )
    popularity_scores = (
        popularity_scores - popularity_scores.mean()
    ) / (popularity_scores.std() + 1e-6)

    results_by_seed: Dict[str, object] = {}
    for seed_count in seed_counts:
        method_rows: Dict[str, List[Dict[str, float]]] = {
            "taste_model": [],
            "popularity": [],
        }
        coverage: Dict[str, Set[int]] = {method: set() for method in method_rows}
        evaluated = 0
        for user_id in users:
            positives = [movie_id for movie_id in histories[user_id] if movie_id in item_rows]
            if len(positives) <= seed_count:
                continue
            seeds = positives[:seed_count]
            truth = set(positives[seed_count:])
            query = item_matrix[[item_rows[movie_id] for movie_id in seeds]].mean(axis=0)
            norm = float(np.linalg.norm(query))
            if norm == 0:
                continue
            query /= norm

            model_scores = item_matrix @ query
            if args.popularity_weight:
                model_scores = model_scores + args.popularity_weight * popularity_scores
            pop_scores = popularity_scores.copy()
            for movie_id in seeds:
                row = item_rows[movie_id]
                model_scores[row] = -np.inf
                pop_scores[row] = -np.inf

            for method, scores in (
                ("taste_model", model_scores),
                ("popularity", pop_scores),
            ):
                ranked = [int(item_ids[row]) for row in top_k_indices(scores, 100)]
                method_rows[method].append(ranking_metrics(ranked, truth))
                coverage[method].update(ranked[:100])
            evaluated += 1

        results_by_seed[str(seed_count)] = {
            "users_evaluated": evaluated,
            "methods": {
                method: mean_metrics(rows)
                | {"catalog_coverage@100": len(coverage[method]) / len(item_ids)}
                for method, rows in method_rows.items()
            },
        }

    metrics = {
        "protocol": "global_future_cold_users_earliest_positive_seeds",
        "cohort": args.cohort,
        "cohort_split_seed": args.split_seed,
        "cohort_counts": cohort_counts,
        "users_selected": len(users),
        "selected_user_sha256": hashlib.sha256(
            ",".join(map(str, users)).encode()
        ).hexdigest(),
        "catalog_size": len(item_ids),
        "cutoff_timestamp": cutoff,
        "cutoff_utc": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
        "val_fraction": args.val_fraction,
        "positive_threshold": args.positive_threshold,
        "popularity_weight": args.popularity_weight,
        "item_embeddings_sha256": file_sha256(item_path),
        "results_by_seed_count": results_by_seed,
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
