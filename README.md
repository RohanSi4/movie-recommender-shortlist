# Shortlist

An end-to-end movie recommendation system: retrieval builds the shortlist,
ranking orders it. Takes ranking work out of a notebook and puts it behind an
app people can actually use.

[Try the live app](https://movie-reccomender-system-red.vercel.app) ·
[Read the case study](https://rohansingh04.com/projects/movie-recommender)

![Searching for a movie](docs/ui.png)

![Ranked recommendations with scores and reasons](docs/recs.png)

## What it does

- Recommends movies for a MovieLens user from their historical ratings
- Filters out movies that user already rated
- Searches by title and finds learned similar movies from a chosen starting point
- Adds TMDB genres, release details, popularity, and posters
- Returns scores and short reasons instead of a bare list of titles
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
  exact dot-product retrieval, optional ranking, search, explanations
                |
                v
Next.js app
  title search, user recommendations, result cards
~~~

The public demo serves the learned two-tower retrieval model directly in Go.
Known users get personalized candidates from the full 87,585-movie catalog,
with their training-window history removed. Movie search uses the same learned
item space for similarity. Users outside the trained vocabulary fall back to
the feature-table popularity heuristic. LightGBM reranking remains optional
and is only used when `MODEL_API_BASE` is configured.

## Run it locally

Requirements: Python 3, Go 1.21+, and Node.js 18+.

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
cd frontend && npm run lint && npm run build
~~~

The evaluation and serving paths stay separate on purpose. Offline metrics show
whether the model learned something useful; the live app shows whether the whole
system is understandable and fast enough to use.

## Main API routes

- `POST /rank` for user-based or movie-based recommendations
- `GET /search?q=matrix&limit=10` for title search
- `GET /movie/{movie_id}` for movie details
- `GET /health` for service health

The live frontend is deployed on Vercel. The Go API is deployed separately so
the site and ranking service can scale and fail independently.
