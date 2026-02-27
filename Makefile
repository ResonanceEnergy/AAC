# ═══════════════════════════════════════════════════════════════
# AAC — Accelerated Arbitrage Corp  •  Makefile
# ═══════════════════════════════════════════════════════════════
SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/python -m pytest

.DEFAULT_GOAL := help

# ── Setup ──────────────────────────────────────────────────────
.PHONY: venv install dev-install

venv:  ## Create virtual environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install: install  ## Install dev/test dependencies + pre-commit
	$(PIP) install pre-commit pytest-timeout pytest-cov
	$(VENV)/bin/pre-commit install

# ── Testing ────────────────────────────────────────────────────
.PHONY: test test-fast test-all test-cov lint typecheck

test:  ## Run tests (skip live/exchange/slow)
	$(PYTEST) tests/ -q --tb=short -m "not live and not exchange and not slow" --timeout=15

test-fast:  ## Run tests without network calls
	$(PYTEST) tests/ -q --tb=line -m "not live and not exchange and not slow" --timeout=10 -x

test-all:  ## Run ALL tests including live
	$(PYTEST) tests/ -v --tb=short --timeout=30

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -q -m "not live and not exchange and not slow" --timeout=15 \
		--cov=shared --cov=core --cov=strategies --cov-report=html --cov-report=term

# ── Code Quality ───────────────────────────────────────────────
lint:  ## Lint with black + isort + flake8
	$(VENV)/bin/black --check src/ shared/ core/ tests/ strategies/ 2>/dev/null || true
	$(VENV)/bin/isort --check src/ shared/ core/ tests/ strategies/ 2>/dev/null || true
	$(VENV)/bin/flake8 src/ shared/ core/ tests/ strategies/ 2>/dev/null || true

typecheck:  ## Run mypy type checking
	$(VENV)/bin/mypy src/aac/ shared/ core/ --ignore-missing-imports 2>/dev/null || true

format:  ## Auto-format code
	$(VENV)/bin/black src/ shared/ core/ tests/ strategies/ 2>/dev/null || true
	$(VENV)/bin/isort src/ shared/ core/ tests/ strategies/ 2>/dev/null || true

# ── Launch ─────────────────────────────────────────────────────
.PHONY: dashboard paper core full monitor

dashboard:  ## Start Dash dashboard on :8050
	./launch.sh dashboard

paper:  ## Start paper trading mode
	./launch.sh paper

core:  ## Start core engine only
	./launch.sh core

full:  ## Start all services
	./launch.sh full

monitor:  ## Start system health monitor
	./launch.sh monitor

# ── Maintenance ────────────────────────────────────────────────
.PHONY: clean clean-all freeze

clean:  ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache

clean-all: clean  ## Deep clean including venv
	rm -rf $(VENV) *.egg-info dist build

freeze:  ## Update requirements-lock.txt
	$(PIP) freeze > requirements-lock.txt
	@echo "Wrote requirements-lock.txt ($$(wc -l < requirements-lock.txt) packages)"

health:  ## Run system health check
	$(PY) scripts/health_check.py

# ── Help ───────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
