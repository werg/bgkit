# Load .env (single source of truth for DATA_DIR, CHECKPOINT_DIR, etc.)
-include .env
export

# Compose command — always pass --env-file so direct invocation also works
# (Makefile export covers `make` targets, but --env-file covers docker compose
# reading the file itself for ${VAR} interpolation in the YAML).
DC := docker compose --env-file .env -f docker/docker-compose.yaml

.PHONY: install install-gpu install-fa4-local install-gpu-local-fa4 test test-unit test-gpu test-integration test-smoke lint format train eval ablation docker-build-deps docker-build-data docker-build-llama process-repos train-phase1-step1 train-phase1-step2 train-phase1-step3 train-phase1-step4 train-phase1-step5 train-phase1-step6 flashqla-smoke flashqla-parity flashqla-profile flashqla-shell ckpt-backfill extract-structural generate-descriptions generate-descriptions-atlas extract-commits prepare-commit-encoding convert-tokens convert-tokens-falcon convert-structural convert-descriptions convert-commits convert-commit-encoding generate-variants generate-qa-pairs convert-qa-pairs download-models download-models-hf llama-server llama-server-stop llama-server-logs llama-bench vllm-server vllm-server-stop vllm-server-logs atlas-server atlas-server-stop atlas-server-logs prepare-data prepare-data-full prepare-data-all

FLASH_ATTN_DIR ?= ../flash-attention

install:
	uv sync --extra dev --extra data

install-gpu:
	uv sync --group torch --extra gpu --extra dev --extra eval

install-fa4-local:
	@test -f "$(FLASH_ATTN_DIR)/flash_attn/cute/pyproject.toml" || { echo "ERROR: FLASH_ATTN_DIR=$(FLASH_ATTN_DIR) does not point to a flash-attention checkout"; exit 1; }
	uv pip install --python .venv/bin/python -e "$(FLASH_ATTN_DIR)/flash_attn/cute[cu13]"

install-gpu-local-fa4: install-gpu install-fa4-local

test: test-unit

test-unit:
	CUDA_VISIBLE_DEVICES='' uv run pytest tests/unit -v -m "not gpu"

test-gpu:
	$(DC) run --rm --no-deps \
		-v $(CURDIR)/tests:/workspace/bgkit/tests:ro \
		-v $(CURDIR)/pyproject.toml:/workspace/bgkit/pyproject.toml:ro \
		--entrypoint pytest train-phase1-step3 tests/unit -v -m gpu --ignore=tests/unit/data $(ARGS)

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

train-phase1-step1:
	scripts/run-train.sh train-phase1-step1

train-phase1-step2:
	scripts/run-train.sh train-phase1-step2

train-phase1-step3:
	scripts/run-train.sh train-phase1-step3

train-phase1-step4:
	scripts/run-train.sh train-phase1-step4

train-phase1-step5:
	scripts/run-train.sh train-phase1-step5

train-phase1-step6:
	scripts/run-train.sh train-phase1-step6

train-phase2-kb-stage-a:
	scripts/run-train.sh train-phase2-kb-stage-a

train-phase2-kb-stage-b:
	scripts/run-train.sh train-phase2-kb-stage-b

docker-build-deps:
	$(DC) build train

docker-build-data:
	docker build -f docker/Dockerfile.data -t bgkit-data:latest .

docker-build-llama:
	$(DC) build llama-large llama-small llama-tiny

flashqla-smoke:
	$(DC) run --rm smoke-flashqla

flashqla-parity:
	$(DC) run --rm parity-flashqla

flashqla-profile:
	$(DC) run --rm profile-flashqla

flashqla-shell:
	$(DC) run --rm shell-flashqla

download-models:
	scripts/download-model.sh LiquidAI/LFM2-8B-A1B-GGUF LFM2-8B-A1B-Q4_K_M.gguf
	scripts/download-model.sh LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q8_0.gguf
	scripts/download-model.sh Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf

llama-server:
	$(DC) up -d llama-large llama-small llama-tiny
	@echo "Waiting for llama-large health..."
	@timeout 120 bash -c 'until curl -sf http://localhost:$${LLAMA_PORT_LARGE:-8080}/health; do sleep 2; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }
	@echo "Waiting for llama-small health..."
	@timeout 120 bash -c 'until curl -sf http://localhost:$${LLAMA_PORT_SMALL:-8081}/health; do sleep 2; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }
	@echo "Waiting for llama-tiny health..."
	@timeout 120 bash -c 'until curl -sf http://localhost:$${LLAMA_PORT_TINY:-8082}/health; do sleep 2; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }

llama-server-stop:
	$(DC) stop llama-large llama-small llama-tiny
	$(DC) rm -f llama-large llama-small llama-tiny

llama-server-logs:
	$(DC) logs -f llama-large llama-small llama-tiny

llama-bench:
	scripts/llama-bench.sh

# --- vLLM inference ---

download-models-hf:
	huggingface-cli download openai/gpt-oss-20b
	huggingface-cli download Qwen/Qwen3.5-0.8B

vllm-server:
	$(DC) up -d vllm-primary
	@echo "Waiting for vllm-primary health..."
	@timeout 300 bash -c 'until curl -sf http://localhost:$${VLLM_PORT_PRIMARY:-8090}/health; do sleep 3; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }
	$(DC) up -d vllm-fast
	@echo "Waiting for vllm-fast health..."
	@timeout 300 bash -c 'until curl -sf http://localhost:$${VLLM_PORT_FAST:-8091}/health; do sleep 3; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }

vllm-server-stop:
	$(DC) stop vllm-primary vllm-fast
	$(DC) rm -f vllm-primary vllm-fast

vllm-server-logs:
	$(DC) logs -f vllm-primary vllm-fast

atlas-server:
	$(DC) up -d atlas
	@echo "Waiting for Atlas health..."
	@timeout 300 bash -c 'until curl -sf http://localhost:$${ATLAS_PORT:-8888}/v1/models; do sleep 3; done' && echo " OK" || { echo " TIMEOUT"; exit 1; }

atlas-server-stop:
	$(DC) stop atlas
	$(DC) rm -f atlas

atlas-server-logs:
	$(DC) logs -f atlas

ckpt-backfill:
	.venv/bin/bgkit-ckpt backfill

# Data preprocessing pipeline (paths from .env via bgkit.env / DATA_DIR)
extract-structural:
	.venv/bin/python scripts/extract_structural_data.py --workers 8 $(ARGS)

generate-descriptions: vllm-server
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/generate_descriptions.py \
		--structural-dir $(DATA_DIR)/structural/ --backend local --workers 2 $(ARGS)

generate-descriptions-atlas: atlas-server
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/generate_descriptions.py \
		--structural-dir $(DATA_DIR)/structural/ --backend atlas --workers 2 $(ARGS)

extract-commits:
	.venv/bin/python scripts/process_commit_repro.py $(ARGS)

prepare-commit-encoding:
	.venv/bin/python scripts/prepare_commit_encoding_data.py $(ARGS)

convert-tokens:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_tokens_to_npy.py \
		--input-dir $(DATA_DIR)/processed/tokens/

convert-tokens-falcon:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_tokens_to_falcon_mmap.py \
		--input-dir $(DATA_DIR)/processed/tokens \
		--output-dir $(DATA_DIR)/processed/tokens_falcon_h1 \
		$(ARGS)

convert-structural:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_structural_to_npy.py \
		--input-dir $(DATA_DIR)/structural/ --output-dir $(DATA_DIR)/mmap/structural/

convert-descriptions:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_descriptions_to_npy.py \
		--input-dir $(DATA_DIR)/descriptions/ --output-dir $(DATA_DIR)/mmap/descriptions/

convert-commits:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_commits_to_npy.py \
		--input-dir $(DATA_DIR)/processed/commit_reproduction/

convert-commit-encoding:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_commit_encoding_to_npy.py \
		--input-dir $(DATA_DIR)/processed/commit_encoding/

prepare-data:
	scripts/prepare-data.sh $(ARGS)

prepare-data-full:
	scripts/prepare-data.sh --with-descriptions $(ARGS)

prepare-data-all:
	scripts/prepare-data.sh --with-descriptions --with-qa $(ARGS)

generate-qa-pairs: vllm-server
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/generate_qa_pairs.py \
		--repos-dir $(DATA_DIR)/repos/ \
		--output-dir $(DATA_DIR)/qa_pairs/ \
		--server-url-primary http://localhost:$${VLLM_PORT_PRIMARY:-8090} \
		--server-url-fast http://localhost:$${VLLM_PORT_FAST:-8091}

convert-qa-pairs:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set — copy .env.example to .env"; exit 1; }
	.venv/bin/python scripts/convert_qa_pairs_to_npy.py \
		--input-dir $(DATA_DIR)/qa_pairs/ \
		--output-dir $(DATA_DIR)/mmap/qa_conditioned/

generate-variants:
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/file_read_repro.yaml --num-variants 40
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/description_gen.yaml --num-variants 40
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/structural_repro.yaml --num-variants 40
	.venv/bin/python scripts/generate_prompt_variants.py \
		--template configs/templates/commit_repro.yaml --num-variants 40
