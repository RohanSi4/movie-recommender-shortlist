# Shortlist

A full movie discovery app built around one simple idea: pick a few movies you
already love and get a shortlist shaped by the overlap in your taste. The model,
Go API, and Next.js product all live in this repository.

[Try the live app](https://movie-reccomender-system-red.vercel.app) ·
[Read the case study](https://rohansingh04.com/projects/movie-recommender)

![Building a personal movie mix](docs/ui.png)

![A personal shortlist with movie details and natural reasons](docs/recs.png)

## What it does

- Builds a taste profile from one to five movies a visitor already loves
- Blends their learned item embeddings and searches the full catalog in one pass
- Recommends movies for an anonymous MovieLens viewer from their historical ratings
- Filters out movies that user already rated
- Searches by title and finds learned similar movies from a chosen starting point
- Adds TMDB genres, release details, ratings, overviews, and posters
- Returns natural reasons instead of exposing raw model scores
- Lets visitors save a shortlist locally, dismiss misses, open details, and ask for a fresh batch
- Serves the current live ranking path through a low-latency Go API

I built the data pipeline, models, API, and web app. The two-tower retriever
reached 0.229 recall@100 against 0.127 for popularity on the same held-out
sample. In a 200-request local benchmark, known-user requests had a 4.3 ms
median and 5.8 ms p95 client round trip. Movie-similarity requests had a 3.5 ms
median. The offline LightGBM ranker also beat the heuristic baseline by 11.8%.
The committed [retrieval evaluation](docs/metrics/retrieval_eval.json) and
[latency results](docs/metrics/retrieval_latency.json) keep those claims
checkable.

## How it fits together

~~~text
MovieLens ratings + TMDB metadata
                |
                v
Python ML pipeline
  feature engineering, two-tower retrieval, LightGBM ranking
                |
                v
Verified serving bundle
  float16 user/movie vectors, seen-movie history, checksums
                |
                v
Go ranking API
  taste-vector blending, exact dot-product retrieval, search, explanations
                |
                v
Next.js app
  favorite picker, personal shortlist, details, saved movies
~~~

The public demo serves the learned two-tower retrieval model directly in Go.
New visitors can choose up to five favorites. The service averages and
normalizes those movie vectors into one temporary taste profile, excludes the
chosen movies, and retrieves a new shortlist from the full catalog. No account
or personal data is needed.

Known users get personalized candidates from the full 87,585-movie catalog,
with their training-window history removed. Movie search uses the same learned
item space for similarity. Users outside the trained vocabulary fall back to
the feature-table popularity heuristic. LightGBM reranking remains optional
and is only used when `MODEL_API_BASE` is configured.

## Run it locally

Requirements: Python 3, Go 1.21+, and Node.js 20.9+.

Install the Python dependencies:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

The repository includes exported feature tables and the verified retrieval
bundle for the Go service.
Start the API:

~~~bash
cd service
go run ./cmd/server
~~~

In another terminal, start the site:

~~~bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
~~~

Open [localhost:3000](http://localhost:3000). The frontend points to
`http://localhost:8080` by default.

## Rebuild the ML pipeline

The full sequence is available through the Makefile:

~~~bash
make ingest
make enrich
make features
make training
make train
make export
make train-retrieval
make metrics-retrieval
make export-retrieval
~~~

TMDB enrichment needs `TMDB_API_KEY`. MovieLens supplies the ratings, tags, and
movie identifiers; TMDB supplies the review-friendly metadata and posters.

To serve LightGBM scores locally, start the model service before the Go API:

~~~bash
uvicorn model_service.app:app --host 0.0.0.0 --port 8090
cd service
MODEL_API_BASE=http://localhost:8090 go run ./cmd/server
~~~

## Check the results

~~~bash
make metrics-eval
make metrics-compare
make metrics-scale
make metrics-latency
make test-service
cd frontend
npm run lint
npm run typecheck
npm run build
npm run test:e2e
~~~

The evaluation and serving paths stay separate on purpose. Offline metrics show
whether the model learned something useful; the live app shows whether the whole
system is understandable and fast enough to use.

## Main API routes

- `POST /rank` for user-based or movie-based recommendations
- `POST /rank` with `movie_ids` to build a temporary multi-movie taste profile
- `exclude_movie_ids` on rank requests to fetch genuinely fresh batches
- `GET /search?q=matrix&limit=10` for title search
- `GET /movie/{movie_id}` for movie details
- `GET /health` for service health

The live frontend is deployed on Vercel. The Go API is deployed separately so
the site and ranking service can scale and fail independently.

Every push now runs Go tests and vetting, Python exporter tests, frontend lint,
TypeScript, the production build, Playwright flows, mobile checks, and Axe
accessibility tests. A weekly workflow also checks the live Vercel and Render
deployments for health and stale product copy.
