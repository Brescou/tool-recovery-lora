# ─── Variables ───────────────────────────────────────────────────────────────

UV := uv
PYTHON := uv run python

# ─── Phony Targets ───────────────────────────────────────────────────────────

.PHONY: help install test eval train dataset lint format clean

.DEFAULT_GOAL := help

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

# ─── Development ─────────────────────────────────────────────────────────────

install: ## Install package + dev extras
	$(UV) sync --extra dev

test: ## Run the test suite
	$(UV) run pytest tests/ -v

eval: ## Run objective smoke eval on data/eval/smoke.jsonl
	$(UV) run python scripts/run_eval.py

train: ## Run training stub (requires train extra + GPU for real runs)
	$(UV) run python scripts/run_train.py

dataset: ## Generate curated train/val + smoke eval JSONL
	$(UV) run python scripts/generate_dataset.py

lint: ## Lint with ruff
	$(UV) run ruff check src tests scripts

format: ## Format with ruff
	$(UV) run ruff format src tests scripts

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
