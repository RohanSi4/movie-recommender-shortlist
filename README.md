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
- Searches by title and finds similar movies from a chosen starting point
- Adds TMDB genres, release details, popularity, and posters
- Returns scores and short reasons instead of a bare list of titles
- Serves the current live ranking path through a low-latency Go API

I built the data pipeline, model, API, and web app. The offline LightGBM ranker
beat the current heuristic baseline by 11.8%. The current Go recommendation path
responds in about 17 milliseconds in the project benchmark.

## How it fits together

~~~text
MovieLens ratings + TMDB metadata
                |
                v
Python ML pipeline
  feature engineering, LightGBM training, offline evaluation
                |
                v
Exported user and movie features
                |
                v
Go ranking API
  candidate generation, ranking, search, explanations
                |
                v
Next.js app
  title search, user recommendations, result cards
~~~

The public demo uses a lightweight heuristic over the exported feature tables.
The trained LightGBM model and a FastAPI inference service are included, but the
live Go service does not pretend to serve that model unless `MODEL_API_BASE` is
configured. Keeping that boundary clear matters more than making the demo sound
more advanced than it is.

## Run it locally

Requirements: Python 3, Go 1.21+, and Node.js 18+.

Install the Python dependencies:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

The repository already includes exported feature tables for the Go service.
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
