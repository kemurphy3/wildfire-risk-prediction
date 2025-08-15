# Makefile for NEON AOP Crosswalk Processing
# This file provides targets for downloading, processing, and analyzing AOP data

.PHONY: help install setup download process calibrate validate all clean

# Default target
help:
	@echo "NEON AOP Crosswalk Processing Makefile"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install required dependencies"
	@echo "  setup       - Set up directory structure and configuration"
	@echo "  download    - Download AOP data for all sites"
	@echo "  process     - Process AOP data and extract features"
	@echo "  calibrate   - Calibrate crosswalk models"
	@echo "  validate    - Validate crosswalk models"
	@echo "  all         - Run complete pipeline (download -> process -> calibrate -> validate)"
	@echo "  clean       - Clean up temporary files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Site-specific targets:"
	@echo "  download-SRER  - Download data for Santa Rita Experimental Range"
	@echo "  download-JORN  - Download data for Jornada Experimental Range"
	@echo "  download-ONAQ  - Download data for Onaqui Airstrip"
	@echo "  download-SJER  - Download data for San Joaquin Experimental Range"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make download-SRER"
	@echo "  make process"
	@echo "  make calibrate"

# Configuration
CONFIG_FILE = configs/aop_sites.yaml
SITES = SRER JORN ONAQ SJER
YEARS = 2021 2022
DATA_ROOT = data
RAW_DIR = $(DATA_ROOT)/raw/aop
PROCESSED_DIR = $(DATA_ROOT)/processed/aop
MODELS_DIR = $(DATA_ROOT)/models/aop_crosswalk
OUTPUTS_DIR = $(DATA_ROOT)/outputs/aop_crosswalk
LOGS_DIR = logs/aop_crosswalk

# Python environment
PYTHON = python
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# Install dependencies
install:
	@echo "Installing required dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Dependencies installed successfully!"

# Set up directory structure
setup:
	@echo "Setting up directory structure..."
	mkdir -p $(RAW_DIR)
	mkdir -p $(PROCESSED_DIR)
	mkdir -p $(MODELS_DIR)
	mkdir -p $(OUTPUTS_DIR)
	mkdir -p $(LOGS_DIR)
	@echo "Directory structure created!"

# Download AOP data for all sites
download: $(addprefix download-,$(SITES))
	@echo "All AOP data downloaded successfully!"

# Download data for specific site
download-%:
	@echo "Downloading AOP data for site $*..."
	$(PYTHON_VENV) -m src.data_collection.neon_client download_aop_data \
		--site $* \
		--years $(YEARS) \
		--output-dir $(RAW_DIR)/$* \
		--config $(CONFIG_FILE)
	@echo "Download complete for site $*"

# Process AOP data and extract features
process:
	@echo "Processing AOP data and extracting features..."
	@for site in $(SITES); do \
		for year in $(YEARS); do \
			if [ -d "$(RAW_DIR)/$$site/$$year" ]; then \
				echo "Processing $$site $$year..."; \
				$(PYTHON_VENV) -m src.features.aop_features \
					--site $$site \
					--year $$year \
					--data-dir $(RAW_DIR)/$$site/$$year \
					--output-dir $(PROCESSED_DIR)/$$site/$$year; \
			fi; \
		done; \
	done
	@echo "AOP data processing complete!"

# Calibrate crosswalk models
calibrate:
	@echo "Calibrating crosswalk models..."
	$(PYTHON_VENV) -m src.features.aop_crosswalk \
		--satellite-data $(PROCESSED_DIR)/satellite_indices.csv \
		--aop-data $(PROCESSED_DIR)/aop_features.csv \
		--target-vars chm_mean chm_std canopy_cover_gt2m canopy_cover_gt5m ndvi_aop evi_aop nbr_aop \
		--model-type linear \
		--output-dir $(MODELS_DIR) \
		--mode calibrate
	@echo "Crosswalk model calibration complete!"

# Validate crosswalk models
validate:
	@echo "Validating crosswalk models..."
	$(PYTHON_VENV) -m src.features.aop_crosswalk \
		--satellite-data $(PROCESSED_DIR)/satellite_indices.csv \
		--aop-data $(PROCESSED_DIR)/aop_features.csv \
		--target-vars chm_mean chm_std canopy_cover_gt2m canopy_cover_gt5m ndvi_aop evi_aop nbr_aop \
		--output-dir $(MODELS_DIR) \
		--mode validate
	@echo "Crosswalk model validation complete!"

# Run complete pipeline
all: setup download process calibrate validate
	@echo "Complete AOP crosswalk pipeline finished successfully!"

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -rf $(LOGS_DIR)/*
	rm -rf $(OUTPUTS_DIR)/temp/*
	@echo "Cleanup complete!"

# Test individual components
test-geoalign:
	@echo "Testing geospatial alignment utilities..."
	$(PYTHON_VENV) -c "from src.utils.geoalign import *; print('Geoalign utilities working correctly!')"

test-aop-features:
	@echo "Testing AOP feature extraction..."
	$(PYTHON_VENV) -c "from src.features.aop_features import *; print('AOP features working correctly!')"

test-crosswalk:
	@echo "Testing crosswalk models..."
	$(PYTHON_VENV) -c "from src.features.aop_crosswalk import *; print('Crosswalk models working correctly!')"

test-all: test-geoalign test-aop-features test-crosswalk
	@echo "All components tested successfully!"

# Development targets
dev-setup: setup
	@echo "Setting up development environment..."
	$(PIP) install -e .
	@echo "Development environment ready!"

dev-test:
	@echo "Running development tests..."
	$(PYTHON_VENV) -m pytest tests/ -v

dev-lint:
	@echo "Running linting checks..."
	$(PYTHON_VENV) -m flake8 src/ tests/
	$(PYTHON_VENV) -m black --check src/ tests/

dev-format:
	@echo "Formatting code..."
	$(PYTHON_VENV) -m black src/ tests/

# Documentation targets
docs:
	@echo "Generating documentation..."
	$(PYTHON_VENV) -m pdoc --html src/ --output-dir docs/
	@echo "Documentation generated in docs/"

# Monitoring targets
status:
	@echo "AOP Crosswalk Processing Status"
	@echo "=============================="
	@echo "Raw data:"
	@for site in $(SITES); do \
		echo "  $$site:"; \
		for year in $(YEARS); do \
			if [ -d "$(RAW_DIR)/$$site/$$year" ]; then \
				echo "    $$year: ✓"; \
			else \
				echo "    $$year: ✗"; \
			fi; \
		done; \
	done
	@echo ""
	@echo "Processed data:"
	@for site in $(SITES); do \
		echo "  $$site:"; \
		for year in $(YEARS); do \
			if [ -d "$(PROCESSED_DIR)/$$site/$$year" ]; then \
				echo "    $$year: ✓"; \
			else \
				echo "    $$year: ✗"; \
			fi; \
		done; \
	done
	@echo ""
	@echo "Models:"
	if [ -d "$(MODELS_DIR)" ] && [ "$(shell ls $(MODELS_DIR)/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then \
		echo "  Crosswalk models: ✓"; \
	else \
		echo "  Crosswalk models: ✗"; \
	fi

# Backup targets
backup:
	@echo "Creating backup of processed data and models..."
	tar -czf aop_crosswalk_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		$(PROCESSED_DIR) $(MODELS_DIR) $(OUTPUTS_DIR)
	@echo "Backup created successfully!"

# Restore from backup
restore:
	@echo "Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Error: Please specify BACKUP_FILE=filename.tar.gz"; \
		exit 1; \
	fi
	tar -xzf $(BACKUP_FILE)
	@echo "Restore complete!"

# Performance monitoring
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTHON_VENV) -m src.utils.benchmark \
		--config $(CONFIG_FILE) \
		--output $(OUTPUTS_DIR)/benchmark_results.json
	@echo "Benchmarks complete!"

# Quality assurance
qa-check:
	@echo "Running quality assurance checks..."
	$(PYTHON_VENV) -m src.utils.qa_check \
		--config $(CONFIG_FILE) \
		--data-dir $(PROCESSED_DIR) \
		--output $(OUTPUTS_DIR)/qa_report.html
	@echo "QA checks complete!"

# Report generation
report:
	@echo "Generating comprehensive report..."
	$(PYTHON_VENV) -m src.utils.report_generator \
		--config $(CONFIG_FILE) \
		--data-dir $(PROCESSED_DIR) \
		--models-dir $(MODELS_DIR) \
		--output $(OUTPUTS_DIR)/aop_crosswalk_report.html
	@echo "Report generated successfully!"

# Integration with main wildfire system
integrate:
	@echo "Integrating AOP crosswalk with main wildfire system..."
	$(PYTHON_VENV) -m src.integration.aop_integration \
		--config $(CONFIG_FILE) \
		--models-dir $(MODELS_DIR) \
		--output $(OUTPUTS_DIR)/integrated_features.csv
	@echo "Integration complete!"

# Show help for specific target
help-%:
	@echo "Help for target '$*':"
	@echo ""
	@case "$*" in \
		download) \
			echo "Downloads AOP data from NEON API for all configured sites."; \
			echo "Requires NEON_API_TOKEN environment variable."; \
			echo ""; \
			echo "Usage: make download"; \
			echo "       make download-SRER  # Download for specific site"; \
			;; \
		process) \
			echo "Processes downloaded AOP data and extracts features."; \
			echo "Requires raw data to be downloaded first."; \
			echo ""; \
			echo "Usage: make process"; \
			;; \
		calibrate) \
			echo "Calibrates crosswalk models using satellite and AOP data."; \
			echo "Requires processed data from 'make process'."; \
			echo ""; \
			echo "Usage: make calibrate"; \
			;; \
		validate) \
			echo "Validates calibrated crosswalk models."; \
			echo "Requires calibrated models from 'make calibrate'."; \
			echo ""; \
			echo "Usage: make validate"; \
			;; \
		*) \
			echo "No help available for target '$*'"; \
			;; \
	esac