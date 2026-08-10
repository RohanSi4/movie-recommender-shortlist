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
- Trains that exact favorite-movie mix to retrieve another movie the viewer liked
- Blends their learned item embeddings and searches the full catalog in one pass
- Recommends movies for an anonymous MovieLens viewer from their historical ratings
- Filters out movies that user already rated
- Searches by title and finds learned similar movies from a chosen starting point
- Adds TMDB genres, release details, ratings, overviews, and posters
- Returns natural reasons instead of exposing raw model scores
- Lets visitors save a shortlist locally, dismiss misses, open details, and ask for a fresh batch
- Serves the current live ranking path through a low-latency Go API

I built the data pipeline, models, API, and web app. On an untouched test cohort
of 7,060 future users, the live five-favorite flow reached 0.331 Recall@100
against 0.228 for popularity, a 45% relative lift, while retrieving from 14.5%
of the evaluated catalog instead of popularity's 0.12%. HitRate@10 was 84.1%
against 73.8%, but that top-line metric is generous for this cohort and is
effectively tied at one supplied favorite. The stored-user retriever reached
0.237 Recall@100 against 0.127 for popularity. In a 200-request local benchmark,
known-user requests had a 4.0 ms median and 5.6 ms p95 client round trip. The
committed
[taste evaluation](docs/metrics/taste_eval_test.json),
[warm-user evaluation](docs/metrics/retrieval_eval.json), and
[latency results](docs/metrics/retrieval_latency.json) keep those claims
checkable.

## How that number was measured

A random train/test split would let the model study a viewer's future and then
predict it, so every number here comes from a single global time cutoff instead.

1. **Freeze time once.** All 32 million ratings are sorted by timestamp and the
   earliest 90% become the training window, which ends 2020-11-05. Nothing after
   that date trains anything: not the embeddings, not the user features, not the
   popularity baseline.
2. **Test on people the model never saw.** The test cohort is restricted to
   users with zero ratings before the cutoff, so they did not exist when
   training ended. 14,214 future users qualify, and a hash of the user id splits
   them into 7,154 validation and 7,060 test users.
3. **Choose the model on validation only.** Every training run and every design
   choice was scored on the validation cohort. The test cohort was scored once,
   at the end, on the selected model.
4. **Simulate the real flow.** A test user's earliest one, three, or five
   4.0-and-up ratings become the seeds, exactly what a visitor picks in the app,
   and everything they rated 4.0 or higher afterwards is the truth set. The
   query is the mean of the seed movie vectors, normalized, which is the same
   construction the Go service uses. No user embedding is involved, so a brand
   new visitor is exactly the case being measured.
5. **Give the baseline the same job.** Popularity ranks by pre-cutoff rating
   counts over the identical catalog, for the identical users, with the same
   seed movies removed from the candidate pool.

Each run stamps the cohort hash and the item-embedding file hash into its JSON,
so any published number traces back to one model and one exact user list.

### Reading the numbers honestly

These test users are prolific: the median truth set is about 50 movies, so
landing one of them in a top 10 is not a hard bar and popularity alone clears it
73.8% of the time. Recall@100 and catalog coverage are therefore the
load-bearing results. HitRate@10 remains useful context, not the headline.

| supplied favorites | HitRate@10 | popularity | Recall@100 | popularity | catalog coverage@100 | popularity |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.1% | 81.9% | 0.282 | 0.250 | 27.0% | 0.12% |
| 3 | 84.7% | 77.2% | 0.319 | 0.237 | 19.1% | 0.12% |
| 5 | 84.1% | 73.8% | 0.331 | 0.228 | 14.5% | 0.12% |

Two things in that table are worth saying out loud:

- **At one favorite the model barely beats popularity on HitRate@10**, 82.1%
  against 81.9%. One movie is a thin query. The gap only opens once there is a
  mix to blend, which is why the product asks for up to five. Even at one seed
  though, the model already wins Recall@100 and it draws its answers from 27% of
  the catalog while popularity draws from 0.12%. Popularity is not really
  competing, it is showing the same blockbusters to everyone and being right
  often enough because those movies are widely liked. The model is finding
  different movies for different people.
- **HitRate@10 is not monotonic in seed count**, peaking at three favorites.
  More seeds pull the query toward the average of a person's taste, which keeps
  improving deep recall (0.282 to 0.319 to 0.331) while costing a little top-10
  sharpness.

One caveat I have not solved: the cohort only includes users who went on to rate
at least six movies 4.0 or higher, so it measures active viewers rather than
someone who rates three movies and leaves.

These are offline retrieval results, not evidence that people prefer the
recommendations. There is no online A/B test, satisfaction measure, or viewing
completion outcome yet. The evaluated bundle contained 87,585 movies; the live
serving bundle has since grown to 89,585, so its additional titles were not part
of these reported quality measurements.

## How it fits together

~~~text
MovieLens ratings + TMDB metadata
                |
                v
Python ML pipeline
  user-balanced training, warm-user retrieval, 1-to-5 favorite taste training
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

The public demo serves the learned retriever directly in Go. During training,
one objective learns stored viewer profiles while a second samples one to five
liked movies, averages them exactly as the product does, and predicts another
liked movie. New visitors therefore use a behavior the model was explicitly
trained for. The service keeps the taste profile temporary, excludes the chosen
movies, and searches the full catalog. No account or personal data is needed.

Known users get personalized candidates from the full 89,585-movie catalog,
with their training-window history removed. User-balanced batches stop highly
active viewers from dominating training, de-duplicated targets make each batch
more useful, and log-Q is calculated for the actual sampler. Movie search uses
the same learned item space for similarity. Users outside the trained vocabulary
fall back to the feature-table popularity heuristic. When a visitor seeds the
shortlist with a release too recent to have a trained embedding, the service
blends in a popularity prior in proportion to how cold the seeds are, so a brand
new movie returns well-liked films instead of noise while warm seeds stay fully
personalized. LightGBM reranking remains optional and is only used when
`MODEL_API_BASE` is configured.

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
make discover-tmdb
make export
make train-retrieval
make metrics-retrieval
make metrics-taste
make train-serving
make export-retrieval
~~~

`make discover-tmdb` (needs `TMDB_API_KEY`) pulls releases newer than the 2023
ratings wall from TMDB so 2024+ films enter the catalog. They cannot have a
learned vector, so the export gives each a genre-centroid cold embedding and zero
training support, which the cold-seed popularity blend then handles. It is
optional: the export steps skip it cleanly when the discovered file is absent.

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
make metrics-retrieval
make metrics-taste
make metrics-scale
make metrics-latency
make test-service
cd frontend
npm run lint
npm run typecheck
npm run build
npm run test:e2e
~~~

The evaluation and serving paths stay separate on purpose. The product-aligned
test freezes the training cutoff, finds users with no pre-cutoff history, uses
their earliest one, three, or five future favorites as seeds, and treats their
later favorites as truth. Validation and test users are split deterministically,
and the test cohort is only used after model selection. Offline metrics show
whether the model learned something useful; the live app shows whether the full
system is understandable and fast enough to use. The bundle the app serves is
retrained on nearly the full timeline so recent releases have embeddings, while
the published metrics stay on the frozen holdout: evaluate on a holdout, deploy
on all the data.

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
