# RAG PDF Q&A Benchmark (Simple, Graph, Hybrid)

This project is a Streamlit app for comparing three retrieval styles over PDF content:

- `SimpleRAG`: FAISS vector retrieval from chunked PDFs.
- `GraphRAG`: Neo4j entity graph retrieval.
- `HybridRAG`: Neo4j graph retrieval + Neo4j vector retrieval.

Each page also includes LLM-based answer evaluation against a user-provided ground truth.

## Current Architecture

The codebase is organized around shared utilities and thin page entrypoints:

- `utils/config.py`: Loads `.env` and exposes typed `AppConfig`.
- `utils/rag_components.py`: Shared chat/embedding builders, answer chain builder, evaluator prompt.
- `utils/graph.py`: Unified graph module with:
  - Neo4j client creation and retry logic
  - Neo4j connectivity checks
  - Entity extraction and structured graph retrieval helpers
  - Cached Streamlit graph runtime resources
- `utils/streamlit_helpers.py`: Shared UI helpers (required env checks, evaluation panel)
- `updateDB.py`: Ingestion pipeline (PDF load -> FAISS index + Neo4j graph docs)
- `pages/*.py`: The three app experiences

## Prerequisites

- Python (3.10+ recommended)
- Neo4j instance (Aura or self-hosted)
- OpenAI API key

## Install And Setup

### Fast Path (Makefile)

1. Bootstrap dependencies and `.env` template:
   - `make setup`
2. Edit `.env` with real credentials.
3. Build indexes from PDFs in `data/`:
   - `make ingest`
4. Run the app:
   - `make run`

One-command workflow:

- `make all`

### Manual Path

1. Create venv:
   - `python3 -m venv .venv`
2. Install deps:
   - `.venv/bin/pip install -r requirements.txt`
3. Copy env template:
   - `cp .env.example .env`
4. Fill `.env` with valid OpenAI + Neo4j values.
5. Ingest:
   - `.venv/bin/python updateDB.py`
6. Start Streamlit:
   - `.venv/bin/streamlit run app.py`

## Environment Variables

All config values are loaded by `utils/config.py`.

Use these keys in `.env`:

- `OPENAI_API_KEY`: API key used for all OpenAI calls.
- `NEO4J_URI`: Neo4j connection URI.
- `NEO4J_USERNAME`: Neo4j auth username.
- `NEO4J_PASSWORD`: Neo4j auth password.

Aura example:

- `NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=<your-password>`

## Data Ingestion Details

`updateDB.py` performs two writes:

1. FAISS build:
   - Loads PDFs from `data/`
   - Splits docs using FAISS chunk settings
   - Saves index to `pages/faiss_index/`
2. Graph build:
   - Loads PDFs again
   - Splits docs using graph chunk settings
   - Uses LLM graph transformer
   - Writes graph documents into Neo4j

If no PDFs are found, ingestion fails fast with a clear error.

## App Pages

- `SimpleRAG` (`pages/simpleRAG.py`):
  - Reads local FAISS index
  - Retrieves top relevant chunks
- `GraphRAG` (`pages/graphRAG.py`):
  - Extracts entities from question
  - Queries Neo4j fulltext index + neighborhood relations
- `HybridRAG` (`pages/hybridRAG.py`):
  - Combines GraphRAG structured context with Neo4j vector similarity context

All three pages use:

- Shared QA prompt and chain composition from `utils/rag_components.py`
- Shared evaluation panel from `utils/streamlit_helpers.py`

## Make Targets

- `make setup`: create venv + install deps + ensure `.env`
- `make ingest`: build FAISS + graph indexes
- `make run`: start Streamlit
- `make all`: ingest + run

## Dependencies

- Main dependency file: `requirements.txt`
