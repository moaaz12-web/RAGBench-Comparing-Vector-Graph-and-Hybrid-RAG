from __future__ import annotations

"""Unified graph module: Neo4j clients, graph retrieval, runtime cache, and health checks."""

import re
import time
from typing import Callable, Iterable, TypeVar

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import AppConfig, get_config
from .rag_components import build_chat_model, build_embeddings

try:
    from langchain_neo4j import Neo4jGraph, Neo4jVector
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.vectorstores import Neo4jVector


# -------------------------
# Neo4j connection helpers
# -------------------------

_MAX_CONNECTION_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1.5
T = TypeVar("T")


def _with_neo4j_retries(factory: Callable[[], T], unknown_error: str) -> T:
    """Retry transient Neo4j client initialization failures."""
    last_error: Exception | None = None
    for attempt in range(_MAX_CONNECTION_ATTEMPTS):
        try:
            return factory()
        except Exception as exc:  # pragma: no cover - runtime connectivity errors
            last_error = exc
            if attempt < _MAX_CONNECTION_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise ValueError(format_neo4j_error(last_error)) from last_error
    raise ValueError(unknown_error)


def create_neo4j_graph(config: AppConfig) -> Neo4jGraph:
    return _with_neo4j_retries(
        lambda: Neo4jGraph(**config.neo4j_connection_kwargs),
        "Unknown Neo4j connection error.",
    )


def create_hybrid_vector_index(config: AppConfig, embeddings) -> Neo4jVector:
    return _with_neo4j_retries(
        lambda: Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
            **config.neo4j_connection_kwargs,
        ),
        "Unknown Neo4j vector connection error.",
    )


def format_neo4j_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if (
        "dns" in lowered
        or "resolve address" in lowered
        or "name or service not known" in lowered
    ):
        return (
            f"{message}\n\n"
            "Neo4j host DNS resolution failed. Check `NEO4J_URI` in `.env` and verify the host resolves on your machine."
        )
    if "credentials" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return (
            f"{message}\n\n"
            "Neo4j authentication failed. Verify `NEO4J_USERNAME` and `NEO4J_PASSWORD`."
        )
    if "url is correct" in lowered:
        return (
            f"{message}\n\n"
            "If this is Aura, use `neo4j+s://<instance>.databases.neo4j.io` in `NEO4J_URI`."
        )
    return message


def check_neo4j_connectivity(config: AppConfig) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    try:
        with driver.session(database=config.neo4j_database) as session:
            session.run("RETURN 1 AS ok").single(strict=True)
    except Exception as exc:  # pragma: no cover - runtime connectivity errors
        raise ValueError(format_neo4j_error(exc)) from exc
    finally:
        driver.close()


def run_neo4j_connectivity_check(config: AppConfig | None = None) -> int:
    cfg = config or get_config()
    if cfg.missing_required_vars:
        print(
            "Missing required environment variable(s): "
            + ", ".join(cfg.missing_required_vars)
        )
        return 1
    try:
        check_neo4j_connectivity(cfg)
    except Exception as exc:  # pragma: no cover - runtime connectivity errors
        print("Neo4j connectivity check failed.")
        print(format_neo4j_error(exc))
        return 1
    print("Neo4j connectivity check passed.")
    return 0


# -------------------------
# Graph retrieval helpers
# -------------------------

_LUCENE_RESERVED = re.compile(r"[+\-!(){}\[\]^\"~*?:\\/|&]")
_INDEX_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_]")


class Entities(BaseModel):
    names: list[str] = Field(
        ...,
        description=(
            "Scientific and technical entities, terms, and concepts in the user question."
        ),
    )


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Extract scientific and technical entities from the user question. "
                "Return concise entity names."
            ),
        ),
        ("human", "Question: {question}"),
    ]
)


def normalize_index_name(index_name: str) -> str:
    normalized = _INDEX_SAFE_CHARS.sub("_", index_name.strip())
    return normalized or "entity"


def generate_full_text_query(text: str) -> str:
    cleaned = _LUCENE_RESERVED.sub(" ", text)
    words = [word for word in cleaned.split() if word]
    if not words:
        return ""
    return " AND ".join(f"{word}~2" for word in words)


def ensure_fulltext_index(graph, index_name: str) -> None:
    safe_index_name = normalize_index_name(index_name)
    graph.query(
        f"CREATE FULLTEXT INDEX {safe_index_name} IF NOT EXISTS "
        "FOR (e:__Entity__) ON EACH [e.id]"
    )


def build_entity_chain(chat_model):
    return ENTITY_EXTRACTION_PROMPT | chat_model.with_structured_output(Entities)


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def structured_retriever(
    *,
    question: str,
    graph,
    entity_chain,
    index_name: str,
    fulltext_limit: int,
    relation_limit: int,
) -> str:
    entities = entity_chain.invoke({"question": question})
    entity_names = [name.strip() for name in entities.names if name.strip()]
    if not entity_names:
        return "No graph entities extracted from the question."

    outputs: list[str] = []
    for entity in entity_names:
        fulltext_query = generate_full_text_query(entity)
        if not fulltext_query:
            continue
        response = graph.query(
            """CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $fulltext_limit})
            YIELD node, score
            CALL (node) {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output
            LIMIT $relation_limit
            """,
            {
                "index_name": normalize_index_name(index_name),
                "query": fulltext_query,
                "fulltext_limit": fulltext_limit,
                "relation_limit": relation_limit,
            },
        )
        outputs.extend(item["output"] for item in response if item.get("output"))

    unique_outputs = _unique_in_order(outputs)
    if not unique_outputs:
        return "No graph context found for the extracted entities."
    return "\n".join(unique_outputs)


# -------------------------
# Streamlit graph runtime
# -------------------------

@st.cache_resource(show_spinner=False)
def get_graph(config: AppConfig) -> Neo4jGraph:
    graph = create_neo4j_graph(config)
    ensure_fulltext_index(graph, config.graph_fulltext_index)
    return graph


@st.cache_resource(show_spinner=False)
def get_entity_chain(config: AppConfig):
    entity_llm = build_chat_model(
        model=config.openai_graph_model,
        temperature=0.0,
        config=config,
    )
    return build_entity_chain(entity_llm)


@st.cache_resource(show_spinner=False)
def get_hybrid_vector_index(config: AppConfig) -> Neo4jVector:
    return create_hybrid_vector_index(config, build_embeddings(config))


__all__ = [
    "Neo4jGraph",
    "Neo4jVector",
    "build_entity_chain",
    "check_neo4j_connectivity",
    "create_hybrid_vector_index",
    "create_neo4j_graph",
    "ensure_fulltext_index",
    "format_neo4j_error",
    "generate_full_text_query",
    "get_entity_chain",
    "get_graph",
    "get_hybrid_vector_index",
    "normalize_index_name",
    "run_neo4j_connectivity_check",
    "structured_retriever",
]
