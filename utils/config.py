from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UTILS_DIR.parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

# Environment keys used by this app:
# - OPENAI_API_KEY: OpenAI API key for embeddings, answering, graph extraction, and evaluation.
# - NEO4J_URI (or legacy NEO4J_URL): Neo4j connection URI.
# - NEO4J_USERNAME: Neo4j username.
# - NEO4J_PASSWORD: Neo4j password.
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_GRAPH_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_EVAL_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_INGEST_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_TEMPERATURE = 0.0

DEFAULT_NEO4J_DATABASE = "neo4j"

DEFAULT_GRAPH_FULLTEXT_INDEX = "entity"
DEFAULT_GRAPH_FULLTEXT_LIMIT = 10
DEFAULT_GRAPH_RELATION_LIMIT = 50
DEFAULT_RAG_TOP_K = 3

DEFAULT_FAISS_CHUNK_SIZE = 1500
DEFAULT_FAISS_CHUNK_OVERLAP = 200
DEFAULT_GRAPH_CHUNK_SIZE = 800
DEFAULT_GRAPH_CHUNK_OVERLAP = 120

DEFAULT_DATA_DIR = (PROJECT_DIR / "data").resolve()
DEFAULT_FAISS_INDEX_DIR = (PROJECT_DIR / "pages/faiss_index").resolve()
DEFAULT_PDF_GLOB = "*.pdf"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _env_first(names: list[str], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    openai_chat_model: str
    openai_graph_model: str
    openai_eval_model: str
    openai_ingest_model: str
    openai_embedding_model: str
    openai_temperature: float
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    graph_fulltext_index: str
    graph_fulltext_limit: int
    graph_relation_limit: int
    rag_top_k: int
    faiss_chunk_size: int
    faiss_chunk_overlap: int
    graph_chunk_size: int
    graph_chunk_overlap: int
    data_dir: Path
    faiss_index_dir: Path
    pdf_glob: str

    @property
    def neo4j_connection_kwargs(self) -> dict[str, str]:
        return {
            "url": self.neo4j_uri,
            "username": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
        }

    @property
    def missing_required_vars(self) -> list[str]:
        missing: list[str] = []
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            missing.append("OPENAI_API_KEY")
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_username:
            missing.append("NEO4J_USERNAME")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        return missing


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig(
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        openai_chat_model=_env_str("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        openai_graph_model=_env_str("OPENAI_GRAPH_MODEL", DEFAULT_OPENAI_GRAPH_MODEL),
        openai_eval_model=_env_str("OPENAI_EVAL_MODEL", DEFAULT_OPENAI_EVAL_MODEL),
        openai_ingest_model=_env_str("OPENAI_INGEST_MODEL", DEFAULT_OPENAI_INGEST_MODEL),
        openai_embedding_model=_env_str(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL
        ),
        openai_temperature=DEFAULT_OPENAI_TEMPERATURE,
        # Accept both legacy NEO4J_URL and preferred NEO4J_URI.
        neo4j_uri=_env_first(["NEO4J_URI", "NEO4J_URL"], ""),
        neo4j_username=_env_str("NEO4J_USERNAME", ""),
        neo4j_password=_env_str("NEO4J_PASSWORD", ""),
        neo4j_database=_env_str("NEO4J_DATABASE", DEFAULT_NEO4J_DATABASE),
        graph_fulltext_index=DEFAULT_GRAPH_FULLTEXT_INDEX,
        graph_fulltext_limit=DEFAULT_GRAPH_FULLTEXT_LIMIT,
        graph_relation_limit=DEFAULT_GRAPH_RELATION_LIMIT,
        rag_top_k=DEFAULT_RAG_TOP_K,
        faiss_chunk_size=DEFAULT_FAISS_CHUNK_SIZE,
        faiss_chunk_overlap=DEFAULT_FAISS_CHUNK_OVERLAP,
        graph_chunk_size=DEFAULT_GRAPH_CHUNK_SIZE,
        graph_chunk_overlap=DEFAULT_GRAPH_CHUNK_OVERLAP,
        data_dir=DEFAULT_DATA_DIR,
        faiss_index_dir=DEFAULT_FAISS_INDEX_DIR,
        pdf_glob=DEFAULT_PDF_GLOB,
    )
