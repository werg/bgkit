.PHONY: install install-gpu test test-unit test-integration test-smoke lint format train eval ablation docker-build-deps docker-build-data docker-build-llama process-repos ice-labels train-ice train-phase1-step1 train-phase1-step2 ckpt-backfill extract-structural generate-descriptions extract-commits convert-structural convert-descriptions convert-commits generate-variants download-models llama-server llama-server-stop llama-server-logs llama-bench

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

train-ice:
	scripts/run-train.sh train-ice

train-phase1-step1:
	scripts/run-train.sh train-phase1-step1

train-phase1-step2:
	scripts/run-train.sh train-phase1-step2

docker-build-deps:
	docker compose -f docker/docker-compose.yaml build train

docker-build-data:
	docker build -f docker/Dockerfile.data -t bgkit-data:latest .

docker-build-llama:
	docker compose -f docker/docker-compose.yaml build llama

download-models:
	scripts/download-model.sh unsloth/GLM-4.7-Flash-GGUF GLM-4.7-Flash-Q4_K_M.gguf

llama-server:
	docker compose -f docker/docker-compose.yaml up -d llama
	@echo "Waiting for llama-server health..."
	@timeout 120 bash -c 'until curl -sf http://localhost:$${LLAMA_PORT:-8080}/health; do sleep 2; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }

llama-server-stop:
	docker compose -f docker/docker-compose.yaml stop llama
	docker compose -f docker/docker-compose.yaml rm -f llama

llama-server-logs:
	docker compose -f docker/docker-compose.yaml logs -f llama

llama-bench:
	scripts/llama-bench.sh

ckpt-backfill:
	.venv/bin/bgkit-ckpt backfill

# Data preprocessing pipeline (host-only, uses local data/ paths)
extract-structural:
	.venv/bin/python scripts/extract_structural_data.py \
		--repos-dir data/repos/ --output-dir data/structural/ --workers 8

generate-descriptions:
	.venv/bin/python scripts/generate_descriptions.py \
		--repos-dir data/repos/ --output-dir data/descriptions/ \
		--structural-dir data/structural/ --backend local

extract-commits:
	.venv/bin/python scripts/process_commit_repro.py $(ARGS)

convert-structural:
	.venv/bin/python scripts/convert_structural_to_npy.py \
		--input-dir data/structural/ --output-dir data/mmap/structural/

convert-descriptions:
	.venv/bin/python scripts/convert_descriptions_to_npy.py \
		--input-dir data/descriptions/ --output-dir data/mmap/descriptions/

convert-commits:
	.venv/bin/python scripts/convert_commits_to_npy.py \
		--input-dir data/processed/commit_reproduction/

generate-variants:
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/file_read_repro.yaml \
		--num-variants 40 --output data/prompt_variants/file_read_repro.json
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/description_gen.yaml \
		--num-variants 40 --output data/prompt_variants/description_gen.json
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/structural_repro.yaml \
		--num-variants 40 --output data/prompt_variants/structural_repro.json
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/commit_repro.yaml \
		--num-variants 40 --output data/prompt_variants/commit_repro.json
