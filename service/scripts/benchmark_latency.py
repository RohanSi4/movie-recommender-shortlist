#!/usr/bin/env python3
"""Benchmark warm retrieval, cold fallback, and movie similarity separately."""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /rank latency.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--users-csv", type=Path, default=Path("service/data/user_features.csv"))
    parser.add_argument("--user-embeddings", type=Path, default=Path("service/data/user_embeddings.bin"))
    parser.add_argument("--item-embeddings", type=Path, default=Path("service/data/item_embeddings.bin"))
    parser.add_argument("--requests", type=int, default=100, help="Measured requests per mode.")
    parser.add_argument("--warmup", type=int, default=20, help="Unmeasured requests per mode.")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument(
        "--mode",
        choices=("all", "known-user", "cold-user", "movie"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def embedding_ids(path: Path) -> list[int]:
    with path.open("rb") as handle:
        header = handle.read(16)
        if len(header) != 16 or header[:4] != b"EMB1":
            raise ValueError(f"{path} is not an EMB1 file")
        count, _, _ = struct.unpack("<III", header[4:])
        raw_ids = handle.read(count * 4)
        if len(raw_ids) != count * 4:
            raise ValueError(f"{path} has a truncated id table")
    return list(struct.unpack(f"<{count}i", raw_ids))


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "p99_ms": round(float(np.percentile(array, 99)), 3),
        "mean_ms": round(float(np.mean(array)), 3),
    }


def benchmark_mode(
    session: requests.Session,
    base_url: str,
    payloads: list[dict[str, int]],
    expected_strategy: str,
    requests_count: int,
    warmup: int,
    k: int,
) -> dict[str, Any]:
    client_timings: list[float] = []
    server_timings: list[float] = []
    for index in range(warmup + requests_count):
        payload = payloads[index % len(payloads)] | {"k": k}
        started = time.perf_counter()
        response = session.post(f"{base_url}/rank", json=payload, timeout=15)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("strategy") != expected_strategy:
            raise RuntimeError(
                f"Expected {expected_strategy}, received {body.get('strategy')} for {payload}"
            )
        if len(body.get("results", [])) != k:
            raise RuntimeError(f"Expected {k} results for {payload}, received {len(body.get('results', []))}")
        if index >= warmup:
            client_timings.append(elapsed_ms)
            server_timings.append(float(body.get("latency_ms", 0)))
    return {
        "requests": requests_count,
        "client_round_trip": percentile_summary(client_timings),
        "server_reported": percentile_summary(server_timings),
    }


def main() -> None:
    args = parse_args()
    if args.requests <= 0 or args.warmup < 0 or not 1 <= args.k <= 100:
        raise ValueError("requests must be positive, warmup nonnegative, and k between 1 and 100")

    rng = random.Random(args.seed)
    known_users = embedding_ids(args.user_embeddings)
    movie_ids = embedding_ids(args.item_embeddings)
    all_users = pd.read_csv(args.users_csv, usecols=["userId"])["userId"].dropna().astype(int).tolist()
    known_set = set(known_users)
    cold_users = [user_id for user_id in all_users if user_id not in known_set]
    if not known_users or not movie_ids or not cold_users:
        raise RuntimeError("Need known users, cold users, and movies to benchmark every serving path")

    rng.shuffle(known_users)
    rng.shuffle(cold_users)
    rng.shuffle(movie_ids)
    sample_size = max(args.requests + args.warmup, 100)
    modes = {
        "known-user": ("two_tower", [{"user_id": value} for value in known_users[:sample_size]]),
        "cold-user": ("popularity_fallback", [{"user_id": value} for value in cold_users[:sample_size]]),
        "movie": ("two_tower_movie_similarity", [{"movie_id": value} for value in movie_ids[:sample_size]]),
    }
    selected = modes if args.mode == "all" else {args.mode: modes[args.mode]}
    session = requests.Session()
    output = {
        name: benchmark_mode(
            session,
            args.base_url,
            payloads,
            strategy,
            args.requests,
            args.warmup,
            args.k,
        )
        for name, (strategy, payloads) in selected.items()
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
