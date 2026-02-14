from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UTILS_DIR.parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)


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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    raw_value = _env_str(name, default)
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve()


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
        openai_chat_model=_env_str("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_graph_model=_env_str("OPENAI_GRAPH_MODEL", "gpt-4.1-mini"),
        openai_eval_model=_env_str("OPENAI_EVAL_MODEL", "gpt-4.1-mini"),
        openai_ingest_model=_env_str("OPENAI_INGEST_MODEL", "gpt-4.1-mini"),
        openai_embedding_model=_env_str(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        ),
        openai_temperature=_env_float("OPENAI_TEMPERATURE", 0.0),
        # Accept both legacy NEO4J_URL and preferred NEO4J_URI.
        neo4j_uri=_env_first(["NEO4J_URI", "NEO4J_URL"], ""),
        neo4j_username=_env_str("NEO4J_USERNAME", ""),
        neo4j_password=_env_str("NEO4J_PASSWORD", ""),
        neo4j_database=_env_str("NEO4J_DATABASE", "neo4j"),
        graph_fulltext_index=_env_str("GRAPH_FULLTEXT_INDEX", "entity"),
        graph_fulltext_limit=_env_int("GRAPH_FULLTEXT_LIMIT", 10),
        graph_relation_limit=_env_int("GRAPH_RELATION_LIMIT", 50),
        rag_top_k=_env_int("RAG_TOP_K", 3),
        faiss_chunk_size=_env_int("FAISS_CHUNK_SIZE", 1500),
        faiss_chunk_overlap=_env_int("FAISS_CHUNK_OVERLAP", 200),
        graph_chunk_size=_env_int("GRAPH_CHUNK_SIZE", 800),
        graph_chunk_overlap=_env_int("GRAPH_CHUNK_OVERLAP", 120),
        data_dir=_env_path("DATA_DIR", "./data"),
        faiss_index_dir=_env_path("FAISS_INDEX_DIR", "./pages/faiss_index"),
        pdf_glob=_env_str("PDF_GLOB", "*.pdf"),
    )
