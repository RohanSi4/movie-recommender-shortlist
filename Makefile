.PHONY: help venv install ingest enrich features training train train-retrieval export export-retrieval model-service service frontend metrics-eval metrics-retrieval metrics-taste metrics-scale metrics-latency metrics-compare test-service test-frontend test

help:
	@echo "make venv            - create + activate venv + install deps (macOS/zsh)"
	@echo "make install         - install python deps"
	@echo "make ingest          - ingest MovieLens CSVs"
	@echo "make enrich          - TMDB enrichment (requires TMDB_API_KEY)"
	@echo "make features        - build feature tables (uses TMDB if present)"
	@echo "make training        - build train/val datasets"
	@echo "make train           - train LightGBM model"
	@echo "make train-retrieval - train the two-tower retrieval model"
	@echo "make export          - export CSVs for Go service"
	@echo "make export-retrieval - export verified retrieval binaries for Go"
	@echo "make model-service   - run FastAPI model server"
	@echo "make service         - run Go API (MODEL_API_BASE optional)"
	@echo "make frontend        - run Next.js UI"
	@echo "make metrics-eval    - offline NDCG model vs baseline"
	@echo "make metrics-retrieval - offline two-tower recall vs baselines"
	@echo "make metrics-taste   - anonymous 1/3/5-favorite test-cohort metrics"
	@echo "make metrics-compare - offline NDCG model vs heuristic"
	@echo "make metrics-scale   - dataset scale stats"
	@echo "make metrics-latency - API latency bench (server must be running)"
	@echo "make test-service    - run Go serving and exporter tests"
	@echo "make test-frontend   - run lint, types, build, Playwright, and Axe"
	@echo "make test            - run every service and frontend check"

venv:
	python3 -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	pip install -r requirements.txt

ingest:
	python ml/scripts/ingest_movielens.py --raw-dir ml/data/raw --out-dir ml/data/processed

enrich:
	python ml/scripts/enrich_tmdb.py --processed-dir ml/data/processed --out ml/data/processed/tmdb_enriched.csv

features:
	python ml/scripts/build_features.py --processed-dir ml/data/processed --tmdb-csv ml/data/processed/tmdb_enriched.csv --out-dir ml/data/processed/features

training:
	python ml/scripts/build_training_dataset.py --processed-dir ml/data/processed --features-dir ml/data/processed/features --out-dir ml/data/processed/training

train:
	python ml/scripts/train_lightgbm.py --training-dir ml/data/processed/training --out-dir ml/models --max-per-user 5000

train-retrieval:
	python ml/scripts/train_two_tower.py --processed-dir ml/data/processed --out-dir ml/models/two_tower_taste --epochs 3 --batch-size 1024 --sampling-strategy user-balanced --taste-loss-weight 0.5

# Serving model trains on nearly the full timeline (tiny holdout) so recent
# releases get real embeddings. Metrics still come from train-retrieval's
# temporal-holdout model: evaluate on a holdout, deploy on all the data.
train-serving:
	python ml/scripts/train_two_tower.py --processed-dir ml/data/processed --out-dir ml/models/two_tower_taste_serving --epochs 3 --batch-size 1024 --sampling-strategy user-balanced --taste-loss-weight 0.5 --val-fraction 0.01

export:
	python ml/scripts/export_service_data.py --features-dir ml/data/processed/features --out-dir service/data

export-retrieval:
	python ml/scripts/export_embeddings.py --model-dir ml/models/two_tower_taste_serving --processed-dir ml/data/processed --out-dir service/data --val-fraction 0.01

model-service:
	uvicorn model_service.app:app --host 0.0.0.0 --port 8090

service:
	cd service && MODEL_API_BASE=$${MODEL_API_BASE} go run ./cmd/server

frontend:
	cd frontend && npm run dev

metrics-eval:
	python ml/scripts/evaluate_model.py --training-dir ml/data/processed/training --model-dir ml/models --ndcg-k 10

metrics-compare:
	python ml/scripts/compare_heuristic_vs_model.py --training-dir ml/data/processed/training --model-dir ml/models --ndcg-k 10

metrics-retrieval:
	python ml/scripts/evaluate_retrieval.py --processed-dir ml/data/processed --model-dir ml/models/two_tower_taste --out docs/metrics/retrieval_eval.json

metrics-taste:
	python ml/scripts/evaluate_taste_retrieval.py --processed-dir ml/data/processed --model-dir ml/models/two_tower_taste --cohort test --max-users 0 --out docs/metrics/taste_eval_test.json

metrics-scale:
	python ml/scripts/report_dataset_stats.py --processed-dir ml/data/processed --features-dir ml/data/processed/features

metrics-latency:
	python service/scripts/benchmark_latency.py --base-url http://localhost:8080 --requests 200 --warmup 20 --k 25 --mode all

test-service:
	cd service && go test ./...
	python -m unittest discover -s ml/tests

test-frontend:
	cd frontend && npm run lint
	cd frontend && npm run typecheck
	cd frontend && npm run build
	cd frontend && npm run test:e2e

test: test-service test-frontend
