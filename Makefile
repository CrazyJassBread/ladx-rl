.PHONY: build rom rom-test unit test env-test validate clean-cache clean-rom

LADX_DIR := ladx-disassembly
PYTHON ?= python

build rom:
	$(MAKE) -C $(LADX_DIR) build

rom-test:
	$(MAKE) -C $(LADX_DIR) test

unit:
	$(PYTHON) -m pytest -m "not rom" -q

test: unit rom-test

env-test: rom rom-test
	LADX_ROM_PATH=$(LADX_DIR)/azle.gbc \
	LADX_SYM_PATH=$(LADX_DIR)/azle.sym \
	LADX_STATE_PATH=save_states/azle.gbc.start.state \
	$(PYTHON) -m pytest -m rom -q

validate: rom rom-test
	$(PYTHON) scripts/verify_states.py
	$(PYTHON) scripts/validate_env.py

clean-cache:
	rm -rf .pytest_cache zelda_env.egg-info
	find zelda_env training scripts examples tests -type d -name __pycache__ -prune -exec rm -rf {} +

clean-rom:
	$(MAKE) -C $(LADX_DIR) clean
