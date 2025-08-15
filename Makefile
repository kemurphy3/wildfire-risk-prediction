# Makefile for NEON AOP wildfire crosswalk pipeline

.PHONY: help all clean test lint format
.PHONY: aop_fetch aop_features aop_crosswalk aop_eval aop_all
.PHONY: satellite_fetch satellite_features
.PHONY: install setup

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
SITES := GRSM SOAP SJER SYCA
YEARS := 2015,2017,2019,2021,2023,2024
AOP_DATA_ROOT ?= ./data/raw/aop
AOP_OUT_ROOT ?= ./data/processed/aop
REPORTS_DIR := ./reports/aop

# Help target
help:
	@echo "NEON AOP Wildfire Crosswalk Pipeline"
	@echo "====================================="
	@echo ""
	@echo "Setup commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Set up directories and environment"
	@echo ""
	@echo "AOP pipeline commands:"
	@echo "  aop_fetch     - Download NEON AOP data for target sites"
	@echo "  aop_features  - Extract features from AOP data"
	@echo "  aop_crosswalk - Train/validate satellite-AOP crosswalk models"
	@echo "  aop_eval      - Run evaluation notebook and generate report"
	@echo "  aop_all       - Run complete AOP pipeline"
	@echo ""
	@echo "Satellite pipeline commands:"
	@echo "  satellite_fetch    - Download satellite data for sites"
	@echo "  satellite_features - Extract satellite features"
	@echo ""
	@echo "Utility commands:"
	@echo "  clean         - Remove intermediate files"
	@echo "  test          - Run unit tests"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black"
	@echo ""
	@echo "Variables:"
	@echo "  SITES='$(SITES)'"
	@echo "  YEARS='$(YEARS)'"
	@echo "  AOP_DATA_ROOT='$(AOP_DATA_ROOT)'"
	@echo "  AOP_OUT_ROOT='$(AOP_OUT_ROOT)'"

# Installation
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup: install
	mkdir -p $(AOP_DATA_ROOT)
	mkdir -p $(AOP_OUT_ROOT)
	mkdir -p $(REPORTS_DIR)
	mkdir -p data/models/aop_crosswalk
	mkdir -p data/intermediate/aop
	mkdir -p logs
	@echo "Setup complete. Don't forget to:"
	@echo "1. Copy .env.example to .env and fill in your API keys"
	@echo "2. Verify NEON API access"

# AOP Data Pipeline
aop_fetch:
	@echo "Fetching NEON AOP data for sites: $(SITES)"
	$(PYTHON) -m src.data_collection.neon_client download-aop \
		--sites $(SITES) \
		--years $(YEARS) \
		--products chm,hyperspectral \
		--output $(AOP_DATA_ROOT)

aop_features: aop_fetch
	@echo "Extracting features from AOP data"
	@for site in $(shell echo $(SITES) | tr ',' ' '); do \
		echo "Processing $$site..."; \
		$(PYTHON) -m src.features.aop_features \
			--site $$site \
			--config configs/aop_sites.yaml \
			--output $(AOP_OUT_ROOT); \
	done

aop_crosswalk: aop_features satellite_features
	@echo "Training crosswalk models"
	$(PYTHON) -m src.features.aop_crosswalk \
		--sites $(SITES) \
		--years $(YEARS) \
		--mode calibrate \
		--model-type linear \
		--output data/models/aop_crosswalk/
	@echo "Validating crosswalk models"
	$(PYTHON) -m src.features.aop_crosswalk \
		--sites $(SITES) \
		--years $(YEARS) \
		--mode validate \
		--output $(REPORTS_DIR)

aop_eval: aop_crosswalk
	@echo "Running evaluation notebook"
	jupyter nbconvert --execute notebooks/04_aop_crosswalk_demo.ipynb \
		--to html \
		--output $(REPORTS_DIR)/crosswalk_demo.html \
		--ExecutePreprocessor.timeout=600

aop_all: aop_fetch aop_features aop_crosswalk aop_eval
	@echo "Complete AOP pipeline finished"

# Satellite Data Pipeline
satellite_fetch:
	@echo "Fetching satellite data for AOP sites"
	$(PYTHON) -m src.data_collection.satellite_client download \
		--sites $(SITES) \
		--years $(YEARS) \
		--sensors sentinel2,landsat8 \
		--output data/raw/satellite

satellite_features: satellite_fetch
	@echo "Extracting satellite features"
	$(PYTHON) -m src.features.fire_features extract-satellite \
		--sites $(SITES) \
		--config configs/aop_sites.yaml \
		--output data/processed/satellite

# Testing and Code Quality
test:
	pytest tests/ -v --cov=src --cov-report=html

test-aop:
	pytest tests/test_aop_features.py tests/test_aop_crosswalk.py -v

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	pylint src/ --disable=C0103,C0114,C0115,C0116

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

# Cleaning
clean:
	rm -rf data/intermediate/aop/
	rm -rf $(REPORTS_DIR)/*.png
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean
	rm -rf $(AOP_OUT_ROOT)/*
	rm -rf data/models/aop_crosswalk/*
	rm -rf $(REPORTS_DIR)/*

# Utilities
check-env:
	@if [ ! -f .env ]; then \
		echo "ERROR: .env file not found. Copy .env.example to .env and configure."; \
		exit 1; \
	fi
	@echo "Environment file found"

logs:
	@mkdir -p logs
	@echo "Logs directory ready"

# Development helpers
dev-notebook:
	jupyter lab --notebook-dir=notebooks/

dev-server:
	$(PYTHON) -m src.api.server --reload

# Docker support (optional)
docker-build:
	docker build -t wildfire-aop .

docker-run:
	docker run -it --rm \
		-v $(PWD):/app \
		-v $(AOP_DATA_ROOT):/data/aop \
		wildfire-aop

# Performance profiling
profile-features:
	$(PYTHON) -m cProfile -o profile_results.prof \
		-m src.features.aop_features --site GRSM --year 2017
	$(PYTHON) -m pstats profile_results.prof

# Data validation
validate-data:
	$(PYTHON) scripts/validate_aop_data.py \
		--sites $(SITES) \
		--check-completeness \
		--check-quality

# Generate documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Continuous Integration helpers
ci-test: lint test

ci-deploy:
	@echo "Deploy step would go here"

.PHONY: check-env logs dev-notebook dev-server docker-build docker-run
.PHONY: profile-features validate-data docs ci-test ci-deploy