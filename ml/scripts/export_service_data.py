#!/usr/bin/env python3
"""Export compact CSVs for the Go service."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


MOVIE_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "rating_mean",
    "rating_count",
    "tmdb_vote_avg",
    "tmdb_popularity",
    "tmdb_genres",
    "tmdb_poster_path",
    "tmdb_overview",
    "tmdb_release_date",
]

USER_COLUMNS = [
    "userId",
    "rating_mean",
    "rating_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export data for Go service.")
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing movie_features.parquet and user_features.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for service CSVs.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Optional cap for number of movies exported.",
    )
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]


def normalize_count_column(df: pd.DataFrame) -> pd.DataFrame:
    """Keep count columns integer-shaped so every serving language reads them safely."""
    normalized = df.copy()
    counts = pd.to_numeric(normalized["rating_count"], errors="coerce")
    fractional = counts.dropna() % 1 != 0
    if fractional.any():
        raise ValueError("rating_count contains non-integer values")
    normalized["rating_count"] = counts.astype("Int64")
    return normalized


def main() -> None:
    args = parse_args()
    movies = load_parquet(args.features_dir / "movie_features.parquet")
    users = load_parquet(args.features_dir / "user_features.parquet")

    if args.max_movies:
        movies = movies.head(args.max_movies)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    movie_out = args.out_dir / "movie_features.csv"
    user_out = args.out_dir / "user_features.csv"

    normalize_count_column(ensure_columns(movies, MOVIE_COLUMNS)).to_csv(movie_out, index=False)
    normalize_count_column(ensure_columns(users, USER_COLUMNS)).to_csv(user_out, index=False)

    print("Done.")
    print(f"  {movie_out}")
    print(f"  {user_out}")


if __name__ == "__main__":
    main()
