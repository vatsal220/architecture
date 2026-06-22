ROOT_DIR := ${PWD}
PY := python3

.PHONY: install
install: # Install runtime + test dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

.PHONY: train
train: # Run the full recommendation pipeline end-to-end (single entry point)
	PYTHONPATH=$(ROOT_DIR)/src $(PY) $(ROOT_DIR)/run_pipeline.py

.PHONY: test-unit
test-unit: # Run unit tests
	PYTHONPATH=$(ROOT_DIR)/src pytest -vv -s $(ROOT_DIR)/tests/unit

.PHONY: test-integration
test-integration: # Run integration tests
	PYTHONPATH=$(ROOT_DIR)/src pytest -vv -s $(ROOT_DIR)/tests/integration

.PHONY: test-e2e
test-e2e: # Run end-to-end tests (executes the full pipeline)
	PYTHONPATH=$(ROOT_DIR)/src pytest -vv -s $(ROOT_DIR)/tests/e2e
