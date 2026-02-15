PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
VENV_STREAMLIT := $(VENV_DIR)/bin/streamlit

.RECIPEPREFIX := >
.PHONY: setup install env ingest run all

$(VENV_PYTHON):
>$(PYTHON) -m venv $(VENV_DIR)

setup: install env

install: $(VENV_PYTHON)
>$(VENV_PIP) install --upgrade pip
>$(VENV_PIP) install -r requirements.txt

env:
>@if [ ! -f .env ]; then cp .env.example .env; fi

ingest: setup
>$(VENV_PYTHON) -c "from utils.graph import run_neo4j_connectivity_check; raise SystemExit(run_neo4j_connectivity_check())"
>$(VENV_PYTHON) updateDB.py

run: setup
>$(VENV_PYTHON) -c "from utils.graph import run_neo4j_connectivity_check; raise SystemExit(run_neo4j_connectivity_check())"
>$(VENV_STREAMLIT) run app.py

all: ingest run
