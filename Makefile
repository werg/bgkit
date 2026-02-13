.PHONY: install install-gpu test test-unit test-integration test-smoke lint format train eval ablation docker-build docker-build-data process-repos ice-labels

install:
	uv sync --extra dev --extra data

install-gpu:
	uv sync --group torch --extra gpu --extra dev --extra eval

test: test-unit

test-unit:
	uv run pytest tests/unit -v

test-integration:
	uv run pytest tests/integration -v -m integration

test-smoke:
	uv run pytest tests/smoke -v -m smoke

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

typecheck:
	uv run mypy src/bgkit/

train:
	uv run python scripts/train.py $(ARGS)

eval:
	uv run python scripts/evaluate.py $(ARGS)

ablation:
	uv run python scripts/run_ablation.py $(ARGS)

quality-gate:
	uv run python scripts/run_quality_gate.py $(ARGS)

profile:
	uv run python scripts/profile_compute.py

process-repos:
	.venv/bin/python scripts/process_repos.py $(ARGS)

ice-labels:
	docker compose -f docker/docker-compose.yaml run --rm ice-labels $(ARGS)

docker-build:
	docker build -f docker/Dockerfile -t bgkit:latest .

docker-build-data:
	docker build -f docker/Dockerfile.data -t bgkit-data:latest .
