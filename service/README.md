# Go Ranking Service

Go service for exact learned retrieval, movie similarity, and optional
LightGBM reranking.

## Data
The service reads CSVs from `service/data/`:
- `movie_features.csv`
- `user_features.csv`

It also loads one checksummed retrieval bundle:
- `item_embeddings.bin`
- `user_embeddings.bin`
- `user_history.bin`
- `retrieval_manifest.json`

The embeddings are float16 on disk and widened to float32 at startup. User
history is delta-varint encoded and stays compressed in memory. If the bundle
is missing, partial, mixed between model runs, or corrupt, the service logs the
problem and keeps serving the heuristic fallback.

Generate these from the ML pipeline:
```bash
python ml/scripts/export_service_data.py \
  --features-dir ml/data/processed/features \
  --out-dir service/data

python ml/scripts/export_embeddings.py \
  --model-dir ml/models/two_tower_logq \
  --processed-dir ml/data/processed \
  --out-dir service/data
```

## Run
```bash
go run ./cmd/server
```

Env vars:
- `PORT` (default 8080)
- `MOVIE_DATA_DIR` (default auto-detected: `service/data` or `data`)
- `MODEL_API_BASE` (optional, e.g. `http://localhost:8090` to use LightGBM inference)
- `CANDIDATE_POOL_SIZE` (default 2000, used before optional LightGBM reranking)
- `MEMORY_LIMIT_MB` (default 384, leaves headroom on a 512 MB instance)
- `CORS_ALLOWED_ORIGINS` (comma-separated production frontend origins; local
  defaults are `http://localhost:3000` and `http://localhost:3001`)

For Render, create a Go Web Service with:
```bash
go build -o app ./cmd/server
./app
```

Set `CORS_ALLOWED_ORIGINS` to the deployed Vercel frontend URL, for example:
```bash
CORS_ALLOWED_ORIGINS=https://movie-recommender-demo.vercel.app
```

## Endpoints
- `POST /rank` -> body `{ "user_id": 123, "k": 25 }` (MovieLens user id)  
  or `{ "movie_id": 550, "k": 25 }` (movie-based similar titles)
- `GET /search?q=matrix&limit=10`
- `GET /movie/{movie_id}`
- `GET /health`

## Latency Bench
Run (with server running):
```bash
python service/scripts/benchmark_latency.py --base-url http://localhost:8080 --requests 200 --k 25
```

The benchmark separates known-user retrieval, cold-user fallback, and movie
similarity, validates the returned strategy and result count, and reports both
server time and client round-trip time.

The cold-start list is computed once when the service loads. It does not rescore
and sort the full catalog on every unknown-user request.
