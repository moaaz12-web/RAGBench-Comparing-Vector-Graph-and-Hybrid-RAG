# RAG PDF Q&A Benchmark (Simple, Graph, Hybrid)

This project is a Streamlit app for comparing three retrieval styles over PDF content:

- `SimpleRAG`: FAISS vector retrieval from chunked PDFs.
- `GraphRAG`: Neo4j entity graph retrieval.
- `HybridRAG`: Neo4j graph retrieval + Neo4j vector retrieval.

Each page also includes LLM-based answer evaluation against a user-provided ground truth.

## Current Architecture

The codebase is organized around shared utilities and thin page entrypoints:

- `utils/config.py`: Loads `.env`, applies defaults, exposes typed `AppConfig`.
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
3. Validate Neo4j connectivity:
   - `make check-neo4j`
4. Build indexes from PDFs in `data/`:
   - `make ingest`
5. Run the app:
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

All config values are read in `utils/config.py` from `.env`.

Required for app startup:

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Common model/runtime variables:

- `OPENAI_CHAT_MODEL`, `OPENAI_GRAPH_MODEL`, `OPENAI_EVAL_MODEL`, `OPENAI_INGEST_MODEL`
- `OPENAI_EMBEDDING_MODEL`, `OPENAI_TEMPERATURE`
- `NEO4J_DATABASE`
- `GRAPH_FULLTEXT_INDEX`, `GRAPH_FULLTEXT_LIMIT`, `GRAPH_RELATION_LIMIT`
- `RAG_TOP_K`

Data/index variables:

- `DATA_DIR`, `PDF_GLOB`
- `FAISS_INDEX_DIR`
- `FAISS_CHUNK_SIZE`, `FAISS_CHUNK_OVERLAP`
- `GRAPH_CHUNK_SIZE`, `GRAPH_CHUNK_OVERLAP`

Example Neo4j Aura values:

- `NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=<your-password>`
- `NEO4J_DATABASE=neo4j`

## Data Ingestion Details

`updateDB.py` performs two writes:

1. FAISS build:
   - Loads PDFs from `DATA_DIR` using `PDF_GLOB`
   - Splits docs using FAISS chunk settings
   - Saves index to `FAISS_INDEX_DIR`
2. Graph build:
   - Loads PDFs again
   - Splits docs using graph chunk settings
   - Uses LLM graph transformer
   - Writes graph documents into Neo4j

If no PDFs are found, ingestion fails fast with a clear error.

## App Pages

- `SimpleRAG` (`pages/simpleRAG.py`):
  - Reads local FAISS index
  - Retrieval with `k=RAG_TOP_K`
- `GraphRAG` (`pages/graphRAG.py`):
  - Extracts entities from question
  - Queries Neo4j fulltext index + neighborhood relations
- `HybridRAG` (`pages/hybridRAG.py`):
  - Combines GraphRAG structured context with Neo4j vector similarity context

All three pages use:

- Shared QA prompt and chain composition from `utils/rag_components.py`
- Shared evaluation panel from `utils/streamlit_helpers.py`

## Make Targets

- `make help`: list commands
- `make sequence`: show recommended order
- `make setup`: create venv + install deps + ensure `.env`
- `make check-neo4j`: run Neo4j connectivity check
- `make ingest`: build FAISS + graph indexes
- `make run`: start Streamlit
- `make bootstrap`: setup + check-neo4j + ingest
- `make all`: bootstrap + run
- `make check`: compile/syntax check for `app.py`, `updateDB.py`, `pages`, `utils`
- `make clean`: remove local venv and Python caches

## Troubleshooting

- `Missing required environment variable(s)`:
  - Fill required values in `.env`.
- `FAISS index directory not found` in SimpleRAG:
  - Run `make ingest` first.
- Neo4j DNS/auth errors:
  - Run `make check-neo4j`.
  - Confirm `NEO4J_URI` format and credentials.
  - For Aura, use `neo4j+s://...`.

## Dependencies

- Main dependency file: `requirements.txt`
- Compatibility shim: `req.txt` (references `requirements.txt`)
