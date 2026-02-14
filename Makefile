PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
VENV_STREAMLIT := $(VENV_DIR)/bin/streamlit

.RECIPEPREFIX := >
.PHONY: help sequence setup install env check-neo4j ingest run bootstrap all check clean

help:
>@echo "Available targets:"
>@echo "  sequence  - Print ordered commands (1,2,3...)"
>@echo "  setup     - Create venv + install dependencies + ensure .env exists"
>@echo "  check-neo4j - Validate Neo4j cloud connectivity from .env"
>@echo "  ingest    - Build FAISS + Neo4j indexes from PDFs"
>@echo "  run       - Start Streamlit app"
>@echo "  bootstrap - setup + check-neo4j + ingest"
>@echo "  all       - bootstrap + run"
>@echo "  check     - Syntax compile check"
>@echo "  clean     - Remove local caches and virtual environment"

sequence:
>@echo "1) make setup"
>@echo "2) make check-neo4j"
>@echo "3) make ingest"
>@echo "4) make run"
>@echo ""
>@echo "One-shot alternative: make all"

$(VENV_PYTHON):
>$(PYTHON) -m venv $(VENV_DIR)

setup: install env

install: $(VENV_PYTHON)
>$(VENV_PIP) install --upgrade pip
>$(VENV_PIP) install -r requirements.txt

env:
>@if [ ! -f .env ]; then cp .env.example .env; fi

check-neo4j: setup
>$(VENV_PYTHON) -c "from utils.graph import run_neo4j_connectivity_check; raise SystemExit(run_neo4j_connectivity_check())"

ingest: setup check-neo4j
>$(VENV_PYTHON) updateDB.py

run: setup check-neo4j
>$(VENV_STREAMLIT) run app.py

bootstrap: setup check-neo4j ingest

all: bootstrap run

check: setup
>$(VENV_PYTHON) -m compileall app.py updateDB.py pages utils

clean:
>rm -rf $(VENV_DIR) __pycache__ pages/__pycache__ utils/__pycache__
