# Training image: runs the pipeline (prepare -> train -> MLflow registry).
#   docker build -t ride-train .
#   docker run --rm -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
#       -v "$PWD/data:/app/data" ride-train
FROM python:3.12-slim

# uv pinned (never latest), copied from its official image
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencies only: this layer stays cached until uv.lock changes
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2) Then the project code
COPY README.md params.yaml dvc.yaml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Do not run as root
RUN useradd --create-home app \
    && mkdir -p data/raw data/processed \
    && chown -R app:app /app
USER app

CMD ["python", "-m", "fraud_detection.train"]
