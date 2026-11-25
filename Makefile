# Shortcuts for the most common commands. `make help` lists them.
# Every Python command goes through `uv run` (no venv activation needed).
-include .env
export

RUN = uv run
IMAGE ?= ride-train
API_IMAGE ?= ride-api
WEB_IMAGE ?= ride-webapp

.PHONY: help install lock up down logs dvc-setup data sample prepare train repro push test lint format docker-build docker-smoke placeholder api webapp serving-build serving-smoke k8s-up k8s-forward k8s-down load clean monitoring alerts-test simulate simulate-drift simulate-concept-drift simulate-unit-bug drift-reference drift feedback-data retrain-check retrain

help:  ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv exactly as described in uv.lock (project + dev tools)
	uv sync --locked

lock:  ## Update uv.lock after editing pyproject.toml (then commit uv.lock)
	uv lock

up:  ## Start the dev stack (MinIO, PostgreSQL, MLflow, Prometheus, Grafana)
	docker compose up -d --build

down:  ## Stop the dev stack (data is kept in Docker volumes)
	docker compose down

logs:  ## Follow the dev stack logs
	docker compose logs -f

dvc-setup:  ## Init DVC with the MinIO remote
	bash scripts/setup_dvc.sh

data:  ## Get the versioned GBG500 data from the DVC remote (MinIO, needs `make up`)
	$(RUN) dvc pull

sample:  ## Generate synthetic rides with the GBG500 schema (no real data needed)
	$(RUN) python -m fraud_detection.data sample --rides 150

prepare:  ## Validate + split the raw data
	$(RUN) python -m fraud_detection.data prepare

train:  ## Train, log to MLflow, register the model (fails if quality gate fails)
	$(RUN) python -m fraud_detection.train

repro:  ## Re-run the DVC pipeline (prepare -> train) when something changed
	$(RUN) dvc repro

push:  ## Push data/pipeline outputs to the DVC remote
	$(RUN) dvc push

test:  ## Run the test suite with coverage
	$(RUN) pytest --cov=fraud_detection --cov-report=term-missing

lint:  ## Lint + check formatting (same as the CI)
	uv lock --check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## Auto-format the code
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

docker-build:  ## Build the training image locally
	docker build -t $(IMAGE):dev .

docker-smoke: docker-build  ## Build + smoke test the image (same as the CI)
	bash scripts/smoke_test.sh $(IMAGE):dev

api:  ## Run the API locally (needs MLFLOW_TRACKING_URI and a registered model, see `make placeholder`)
	PREDICTION_LOG_PATH=logs/predictions.jsonl RIDE_STORE_DIR=logs/rides $(RUN) uvicorn fraud_detection.api.main:app --reload --reload-dir src --port 8000

# --- Monitoring (docs/monitoring.md) ---
N ?= 200
LAST ?= 200
SIMULATE = $(RUN) python -m fraud_detection.monitoring.simulate --n $(N)

monitoring:  ## Start Prometheus (:9090) and Grafana (:3000) only
	docker compose up -d prometheus grafana

alerts-test:  ## Unit tests of the Prometheus alert rules (needs Docker)
	docker run --rm -v "$(CURDIR)/infra/prometheus:/rules:ro" -w /rules --entrypoint promtool prom/prometheus:v3.5.0 test rules alerts_test.yml

simulate:  ## Normal traffic against the local API (N=200)
	$(SIMULATE)

simulate-drift:  ## Data drift: rougher rides
	$(SIMULATE) --scenario data-drift

simulate-concept-drift:  ## Concept drift: same signals, other labels
	$(SIMULATE) --scenario concept-drift

simulate-unit-bug:  ## Upstream bug: accelerometer sent in g
	$(SIMULATE) --scenario unit-bug

drift-reference:  ## Build the healthy reference (after `make repro`; needs the champion model)
	$(RUN) python -m fraud_detection.monitoring.drift reference

LOG ?= logs/predictions.jsonl
REPORT ?= reports/drift

drift:  ## Drift verdict (JSON) + Evidently report (HTML) on the last LAST=200 predictions of LOG (exit 3 = drift)
	$(RUN) python -m fraud_detection.monitoring.drift check --log $(LOG) --last $(LAST) --out $(REPORT).json; code=$$?; \
	DO_NOT_TRACK=1 uv run --script scripts/evidently_report.py --log $(LOG) --last $(LAST) --out $(REPORT).html || echo "Evidently report not produced"; \
	exit $$code

# --- Retraining (docs/retraining.md) ---
feedback-data:  ## Collect the labelled production rides into data/feedback
	$(RUN) python -m fraud_detection.retraining.collect

retrain-check:  ## Should we retrain? Decision only, nothing is started
	$(RUN) python -m fraud_detection.retraining.trigger --last $(LAST)

retrain:  ## Same, and start train.yml on GitHub if the answer is yes
	$(RUN) python -m fraud_detection.retraining.trigger --last $(LAST) --execute

placeholder:  ## Register the placeholder ride classifier in MLflow (until the real model is trained)
	$(RUN) python -m fraud_detection.api.dummy_model

webapp:  ## Run the web app locally (API on :8000)
	API_URL=http://localhost:8000 $(RUN) streamlit run webapp/app.py

load:  ## Load test of the API (20 users, 1 min), report in reports/
	mkdir -p reports && $(RUN) locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 1m --csv reports/load_local

serving-build:  ## Build the API and web app images locally
	docker build -f docker/api/Dockerfile -t $(API_IMAGE):dev .
	docker build -f docker/webapp/Dockerfile -t $(WEB_IMAGE):dev .

serving-smoke: serving-build  ## Build + smoke test of the serving images (same as the CI)
	bash scripts/smoke_test_serving.sh $(API_IMAGE):dev $(WEB_IMAGE):dev

k8s-up: serving-build  ## Local cluster (kind), load the images and deploy the stack (needs `make up` + a registered model)
	kind get clusters | grep -qx ride || kind create cluster --name ride --config k8s/kind-config.yaml
	kind load docker-image $(API_IMAGE):dev $(WEB_IMAGE):dev --name ride
	bash scripts/deploy_k8s.sh $(API_IMAGE):dev $(WEB_IMAGE):dev

k8s-forward:  ## Docker Desktop Kubernetes: API on :8080, web app on :8501 (Ctrl-C to stop)
	kubectl -n ride port-forward svc/ride-api 8080:80 & kubectl -n ride port-forward svc/ride-webapp 8501:8501

k8s-down:  ## Delete the local cluster
	kind delete cluster --name ride

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov mlruns coverage.xml junit.xml
	find . -name __pycache__ -prune -exec rm -rf {} +
