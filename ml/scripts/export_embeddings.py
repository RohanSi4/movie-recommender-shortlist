#!/usr/bin/env python3
"""Export two-tower embeddings as compact binaries for the Go service.

Format decision: service/data is committed to git, so the export favors
size over readability. Each file is little-endian:

    magic   4 bytes  b"EMB1"
    count   uint32   number of rows
    dim     uint32   embedding dimension
    dtype   uint32   bytes per value (2 = float16, 4 = float32)
    ids     count * int32
    values  count * dim * dtype, row-major, same order as ids

float16 halves the footprint (items ~11 MB, users ~24 MB vs 48 MB float32)
and the quantization error is far below what changes a dot-product ranking
at 64 dimensions. The Go loader widens back to float32 in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAGIC = b"EMB1"
HISTORY_MAGIC = b"HST1"
STATS_MAGIC = b"STA1"
MANIFEST_NAME = "retrieval_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embeddings for Go service.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing item_embeddings.parquet and user_embeddings.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the .bin files (usually service/data).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing ratings.parquet for the train-history exclusion index.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Most-recent ratings held out during training.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Minimum rating counted as a positive when measuring item support.",
    )
    parser.add_argument(
        "--extra-items",
        type=Path,
        default=None,
        help="Optional CSV of discovered movies (from discover_tmdb.py) to add "
        "with genre-centroid cold-start embeddings.",
    )
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    indexed = []
    for column in frame.columns:
        if column.startswith("e") and column[1:].isdigit():
            indexed.append((int(column[1:]), column))
    indexed.sort()
    expected = list(range(len(indexed)))
    actual = [index for index, _ in indexed]
    if not indexed or actual != expected:
        raise ValueError(f"Embedding columns must be contiguous e0..eN; found {actual[:10]}")
    return [column for _, column in indexed]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int32_ids(frame: pd.DataFrame, id_col: str) -> np.ndarray:
    numeric = pd.to_numeric(frame[id_col], errors="raise").to_numpy()
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{id_col} must contain finite integer ids")
    bounds = np.iinfo(np.int32)
    if numeric.min(initial=0) < bounds.min or numeric.max(initial=0) > bounds.max:
        raise ValueError(f"{id_col} contains an id outside int32 range")
    return numeric.astype(np.int32)


def write_embeddings(out_path: Path, id_col: str, frame: pd.DataFrame) -> dict[str, Any]:
    if id_col not in frame.columns:
        raise ValueError(f"Missing id column {id_col}")
    dims = embedding_columns(frame)
    ids = int32_ids(frame, id_col)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{id_col} contains duplicate ids")
    values32 = frame[dims].to_numpy(dtype=np.float32)
    if not np.isfinite(values32).all():
        raise ValueError(f"{out_path.name} contains non-finite values")
    values = values32.astype(np.float16)
    if not np.isfinite(values).all():
        raise ValueError(f"{out_path.name} overflowed while converting to float16")

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temp_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<III", len(ids), len(dims), values.dtype.itemsize))
        handle.write(ids.astype("<i4").tobytes())
        handle.write(values.astype("<f2").tobytes())
    temp_path.replace(out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({len(ids)} rows x {len(dims)} dims, {size_mb:.1f} MB)")
    return {
        "name": out_path.name,
        "sha256": sha256_file(out_path),
        "count": len(ids),
        "dim": len(dims),
        "dtype": "float16",
    }


def append_uvarint(buffer: bytearray, value: int) -> None:
    while value >= 0x80:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    buffer.append(value)


def write_history(
    out_path: Path,
    ratings: pd.DataFrame,
    warm_user_ids: np.ndarray,
    val_fraction: float,
) -> dict[str, Any]:
    if not 0 < val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    required = {"userId", "movieId", "timestamp"}
    if missing := required - set(ratings.columns):
        raise ValueError(f"ratings.parquet is missing columns: {sorted(missing)}")

    if len(np.unique(warm_user_ids)) != len(warm_user_ids):
        raise ValueError("warm user ids contain duplicates")
    warm_user_ids = np.sort(warm_user_ids)

    split_idx = int(len(ratings) * (1.0 - val_fraction))
    train = ratings.sort_values("timestamp", kind="stable").iloc[:split_idx]
    pairs = train.loc[train["userId"].isin(warm_user_ids), ["userId", "movieId"]]
    pairs = pairs.drop_duplicates().sort_values(["userId", "movieId"], kind="stable")
    pair_users = int32_ids(pairs, "userId")
    pair_movies = int32_ids(pairs, "movieId")

    payload = bytearray()
    offsets = np.zeros(len(warm_user_ids) + 1, dtype="<u8")
    pair_pos = 0
    for row, user_id in enumerate(warm_user_ids):
        previous = 0
        while pair_pos < len(pair_users) and pair_users[pair_pos] == user_id:
            movie_id = int(pair_movies[pair_pos])
            delta = movie_id - previous
            if delta <= 0:
                raise ValueError(f"movie ids for user {user_id} are not strictly increasing")
            append_uvarint(payload, delta)
            previous = movie_id
            pair_pos += 1
        offsets[row + 1] = len(payload)
    if pair_pos != len(pair_users):
        raise ValueError("history contains a user without an embedding")

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        handle.write(HISTORY_MAGIC)
        handle.write(struct.pack("<I", len(warm_user_ids)))
        handle.write(warm_user_ids.astype("<i4").tobytes())
        handle.write(offsets.tobytes())
        handle.write(payload)
    temp_path.replace(out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({len(warm_user_ids)} users, {len(pair_users)} seen pairs, {size_mb:.1f} MB)")
    return {
        "name": out_path.name,
        "sha256": sha256_file(out_path),
        "count": len(warm_user_ids),
        "pairs": len(pair_users),
        "encoding": "delta-uvarint",
    }


def write_item_stats(
    out_path: Path,
    item_ids: np.ndarray,
    ratings: pd.DataFrame,
    val_fraction: float,
    positive_threshold: float,
) -> dict[str, Any]:
    """Train-window positive count per item, aligned to the item embedding ids.

    The Go service turns these counts into a normalized popularity score (for
    the cold-seed blend) and a per-seed warmth signal (how much collaborative
    signal actually trained that item's embedding). One count is the single
    source of truth for both.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    split_idx = int(len(ratings) * (1.0 - val_fraction))
    train = ratings.sort_values("timestamp", kind="stable").iloc[:split_idx]
    positives = train[train["rating"] >= positive_threshold]
    counts_by_movie = positives["movieId"].value_counts()
    counts = np.array(
        [int(counts_by_movie.get(int(mid), 0)) for mid in item_ids], dtype=np.uint32
    )

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        handle.write(STATS_MAGIC)
        handle.write(struct.pack("<I", len(item_ids)))
        handle.write(item_ids.astype("<i4").tobytes())
        handle.write(counts.astype("<u4").tobytes())
    temp_path.replace(out_path)

    warm = int((counts > 0).sum())
    print(
        f"Wrote {out_path} ({len(item_ids)} items, {warm} with training support, "
        f"max {int(counts.max(initial=0))})"
    )
    return {
        "name": out_path.name,
        "sha256": sha256_file(out_path),
        "count": len(item_ids),
        "items_with_support": warm,
        "positive_threshold": positive_threshold,
    }


NO_GENRES = "(no genres listed)"


def genre_centroids(
    items: pd.DataFrame, id_col: str, dims: list[str], genres_by_movie: dict[int, str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Unit-norm mean trained embedding per MovieLens genre, plus a global mean.

    A movie released after the ratings wall has no learned vector. Averaging the
    centroids of the genres it shares with the trained catalog places it in the
    right region of the space, which is enough to seed a taste query and to be
    retrieved. It is the poor-man's content tower until an explicit one exists.
    """
    vectors = items[dims].to_numpy(dtype=np.float64)
    ids = items[id_col].to_numpy()
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row, movie_id in enumerate(ids):
        genres = genres_by_movie.get(int(movie_id))
        if not genres or genres == NO_GENRES:
            continue
        for token in str(genres).split("|"):
            if not token:
                continue
            if token not in sums:
                sums[token] = np.zeros(len(dims))
                counts[token] = 0
            sums[token] += vectors[row]
            counts[token] += 1

    centroids: dict[str, np.ndarray] = {}
    for token, total in sums.items():
        centroids[token] = _unit(total / counts[token])
    return centroids, _unit(vectors.mean(axis=0))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cold_embeddings(
    extra: pd.DataFrame,
    id_col: str,
    dims: list[str],
    centroids: dict[str, np.ndarray],
    global_centroid: np.ndarray,
) -> pd.DataFrame:
    """Genre-centroid embeddings for discovered movies, on the unit sphere."""
    rows: list[list[float]] = []
    for _, record in extra.iterrows():
        matched = [
            centroids[token]
            for token in str(record.get("genres") or "").split("|")
            if token in centroids
        ]
        vector = _unit(np.mean(matched, axis=0)) if matched else global_centroid
        rows.append([int(record[id_col]), *vector.tolist()])
    return pd.DataFrame(rows, columns=[id_col, *dims])


def add_cold_start_items(
    items: pd.DataFrame, extra_path: Path, movies_path: Path
) -> pd.DataFrame:
    """Append discovered movies to the item embeddings with cold-start vectors."""
    if not extra_path.exists():
        print(f"Skipping extra items: {extra_path} not found")
        return items
    dims = embedding_columns(items)
    extra = pd.read_csv(extra_path)
    known = set(items["movieId"].astype(int))
    extra = extra[~extra["movieId"].astype(int).isin(known)].copy()
    if extra.empty:
        print(f"No new items in {extra_path}")
        return items[["movieId", *dims]]

    movies = load_parquet(movies_path)
    genres_by_movie = dict(zip(movies["movieId"].astype(int), movies["genres"]))
    centroids, global_centroid = genre_centroids(items, "movieId", dims, genres_by_movie)
    cold = cold_embeddings(extra, "movieId", dims, centroids, global_centroid)
    combined = pd.concat([items[["movieId", *dims]], cold], ignore_index=True)
    print(f"Added {len(cold)} cold-start items from {extra_path}")
    return combined


def write_manifest(out_dir: Path, files: dict[str, dict[str, Any]], val_fraction: float) -> None:
    run_digest = hashlib.sha256()
    for key in sorted(files):
        run_digest.update(files[key]["sha256"].encode())
    manifest = {
        "format_version": 1,
        "model_run": run_digest.hexdigest()[:16],
        "val_fraction": val_fraction,
        "seen_policy": "all_ratings_in_training_window",
        "files": files,
    }
    out_path = out_dir / MANIFEST_NAME
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_path.replace(out_path)
    print(f"Wrote {out_path} (model run {manifest['model_run']})")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = load_parquet(args.model_dir / "item_embeddings.parquet")
    users = load_parquet(args.model_dir / "user_embeddings.parquet")

    if args.extra_items is not None:
        items = add_cold_start_items(
            items, args.extra_items, args.processed_dir / "movies.parquet"
        )

    user_ids = int32_ids(users, "userId")
    item_ids = int32_ids(items, "movieId")
    ratings = load_parquet(args.processed_dir / "ratings.parquet")
    files = {
        "items": write_embeddings(args.out_dir / "item_embeddings.bin", "movieId", items),
        "users": write_embeddings(args.out_dir / "user_embeddings.bin", "userId", users),
        "history": write_history(
            args.out_dir / "user_history.bin",
            ratings,
            user_ids,
            args.val_fraction,
        ),
        "item_stats": write_item_stats(
            args.out_dir / "item_stats.bin",
            item_ids,
            ratings,
            args.val_fraction,
            args.positive_threshold,
        ),
    }
    write_manifest(args.out_dir, files, args.val_fraction)


if __name__ == "__main__":
    main()
