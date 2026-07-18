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

    user_ids = int32_ids(users, "userId")
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
    }
    write_manifest(args.out_dir, files, args.val_fraction)


if __name__ == "__main__":
    main()
