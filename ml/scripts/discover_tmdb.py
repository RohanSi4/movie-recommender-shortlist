#!/usr/bin/env python3
"""Discover recent movies from TMDB to extend the catalog past the ratings wall.

MovieLens ratings stop in 2023, so films released after it never enter the
two-tower catalog and a visitor cannot even seed the shortlist with them. This
pulls recent releases straight from TMDB's /discover/movie endpoint and writes
them in the same shape the Go service already consumes, keyed by a synthetic
movieId so they never collide with MovieLens ids.

The export steps do the rest: export_embeddings.py gives each discovered movie a
genre-centroid cold embedding (so it works as a taste seed and a candidate) and
zero training support (so the cold-seed popularity blend, not a noisy learned
vector, governs its ranking).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Discovered movies get ids far above the MovieLens range (max ~292,757) so the
# two id spaces never overlap. movieId = SYNTHETIC_ID_BASE + tmdbId keeps the
# mapping deterministic and idempotent across re-runs.
SYNTHETIC_ID_BASE = 100_000_000

# TMDB movie genre ids are stable, so hardcoding the map avoids an extra API
# round trip and a fragile name-string join.
TMDB_GENRE_NAMES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}

# Map TMDB genres onto the MovieLens vocabulary the trained embeddings are
# tagged with, so a cold movie's embedding averages the right genre centroids.
# History and TV Movie have no MovieLens equivalent and are dropped.
TMDB_TO_ML_GENRE = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Children",
    14: "Fantasy", 27: "Horror", 10402: "Musical", 9648: "Mystery",
    10749: "Romance", 878: "Sci-Fi", 53: "Thriller", 10752: "War",
    37: "Western",
}

NO_GENRES = "(no genres listed)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover recent movies from TMDB.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/data/processed/tmdb_discovered.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--start-date",
        default="2023-10-01",
        help="Earliest primary release date to include (just past the ratings wall).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Latest primary release date to include (default: no upper bound).",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=50,
        help="Minimum TMDB vote_count, to drop unrated noise.",
    )
    parser.add_argument(
        "--sort-by",
        default="popularity.desc",
        help="TMDB discover sort order.",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Metadata language.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Pages to fetch (20 movies each; TMDB caps discover at 500).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=4.0,
        help="Max requests per second.",
    )
    parser.add_argument(
        "--synthetic-base",
        type=int,
        default=SYNTHETIC_ID_BASE,
        help="Offset added to tmdbId to form the synthetic movieId.",
    )
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY not set in environment.")
    return api_key


def ml_genres(genre_ids: List[int]) -> str:
    """MovieLens-vocabulary genres for a discovered movie, pipe-delimited."""
    mapped: List[str] = []
    for gid in genre_ids:
        name = TMDB_TO_ML_GENRE.get(gid)
        if name and name not in mapped:
            mapped.append(name)
    return "|".join(mapped) if mapped else NO_GENRES


def tmdb_genre_names(genre_ids: List[int]) -> str:
    """Human-readable TMDB genre names for display, pipe-delimited."""
    names = [TMDB_GENRE_NAMES[gid] for gid in genre_ids if gid in TMDB_GENRE_NAMES]
    return "|".join(names)


def build_row(result: Dict[str, Any], synthetic_base: int = SYNTHETIC_ID_BASE) -> Optional[Dict[str, Any]]:
    """Turn one TMDB discover result into a catalog row, or None if unusable."""
    tmdb_id = result.get("id")
    if tmdb_id is None:
        return None
    genre_ids = [int(g) for g in (result.get("genre_ids") or [])]
    title = result.get("title") or result.get("original_title")
    if not title:
        return None
    return {
        "movieId": synthetic_base + int(tmdb_id),
        "tmdbId": int(tmdb_id),
        "title": title,
        "genres": ml_genres(genre_ids),
        "rating_mean": pd.NA,
        "rating_count": 0,
        "tmdb_vote_avg": result.get("vote_average"),
        "tmdb_vote_count": result.get("vote_count"),
        "tmdb_popularity": result.get("popularity"),
        "tmdb_genres": tmdb_genre_names(genre_ids),
        "tmdb_poster_path": result.get("poster_path"),
        "tmdb_overview": result.get("overview"),
        "tmdb_release_date": result.get("release_date"),
    }


def fetch_page(api_key: str, params: Dict[str, Any], page: int) -> Dict[str, Any]:
    query = dict(params)
    query["page"] = page
    query["api_key"] = api_key
    resp = requests.get(f"{TMDB_BASE_URL}/discover/movie", params=query, timeout=15)
    resp.raise_for_status()
    return resp.json()


def discover(
    api_key: str,
    start_date: str,
    end_date: Optional[str],
    min_votes: int,
    sort_by: str,
    language: str,
    max_pages: int,
    rate_limit: float,
    synthetic_base: int,
) -> pd.DataFrame:
    params: Dict[str, Any] = {
        "sort_by": sort_by,
        "include_adult": "false",
        "include_video": "false",
        "language": language,
        "vote_count.gte": min_votes,
        "primary_release_date.gte": start_date,
    }
    if end_date:
        params["primary_release_date.lte"] = end_date

    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    sleep_s = 1.0 / max(rate_limit, 0.1)

    first = fetch_page(api_key, params, 1)
    total_pages = min(int(first.get("total_pages", 1)), max_pages, 500)
    print(
        f"TMDB reports {int(first.get('total_results', 0)):,} movies since {start_date}; "
        f"fetching {total_pages} page(s)..."
    )
    for page in range(1, total_pages + 1):
        payload = first if page == 1 else fetch_page(api_key, params, page)
        for result in payload.get("results", []):
            row = build_row(result, synthetic_base)
            if row is None or row["tmdbId"] in seen:
                continue
            seen.add(row["tmdbId"])
            rows.append(row)
        if page < total_pages:
            time.sleep(sleep_s)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    api_key = require_api_key()

    frame = discover(
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        min_votes=args.min_votes,
        sort_by=args.sort_by,
        language=args.language,
        max_pages=args.max_pages,
        rate_limit=args.rate_limit,
        synthetic_base=args.synthetic_base,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(frame):,} movies).")


if __name__ == "__main__":
    main()
