from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from .config import AppConfig, get_config


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise question-answering assistant.
Use only the provided context. If the context is insufficient, say so explicitly.
Context:
{context}

Question:
{question}

Answer:"""
)

EVALUATION_PROMPT = ChatPromptTemplate.from_template(
    """You are evaluating a retrieval QA answer.
Rate the predicted answer from 0 to 10 by contextual correctness against the actual answer.

Question:
{question}

Actual Answer:
{actual_answer}

Predicted Answer:
{predicted_answer}"""
)


class EvaluationResult(BaseModel):
    rating: float = Field(..., ge=0, le=10)
    justification: str = Field(..., min_length=1)


def build_chat_model(
    model: str | None = None,
    *,
    temperature: float | None = None,
    config: AppConfig | None = None,
) -> ChatOpenAI:
    cfg = config or get_config()
    kwargs: dict[str, object] = {
        "model": model or cfg.openai_chat_model,
        "temperature": cfg.openai_temperature if temperature is None else temperature,
    }
    if cfg.openai_api_key:
        kwargs["api_key"] = cfg.openai_api_key
    return ChatOpenAI(**kwargs)


def build_embeddings(config: AppConfig | None = None) -> OpenAIEmbeddings:
    cfg = config or get_config()
    kwargs: dict[str, object] = {"model": cfg.openai_embedding_model}
    if cfg.openai_api_key:
        kwargs["api_key"] = cfg.openai_api_key
    return OpenAIEmbeddings(**kwargs)


def format_documents(documents: Sequence[Document]) -> str:
    if not documents:
        return "No supporting context found."

    formatted_chunks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown-source")
        page = doc.metadata.get("page")
        source_tag = f"{source}#page={page}" if page is not None else str(source)
        formatted_chunks.append(f"[Document {idx} | {source_tag}]\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def build_answer_chain(
    *,
    context_runnable: Runnable,
    model: str | None = None,
    config: AppConfig | None = None,
) -> Runnable:
    cfg = config or get_config()
    llm = build_chat_model(model=model, config=cfg)
    return (
        RunnableParallel(
            {
                "context": context_runnable,
                "question": RunnablePassthrough(),
            }
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )


def evaluate_answer(
    question: str,
    actual_answer: str,
    predicted_answer: str,
    *,
    config: AppConfig | None = None,
) -> dict[str, object]:
    cfg = config or get_config()
    evaluator = build_chat_model(
        model=cfg.openai_eval_model,
        temperature=0.0,
        config=cfg,
    ).with_structured_output(EvaluationResult)

    chain = EVALUATION_PROMPT | evaluator
    result = chain.invoke(
        {
            "question": question,
            "actual_answer": actual_answer,
            "predicted_answer": predicted_answer,
        }
    )
    if isinstance(result, EvaluationResult):
        return result.model_dump()
    return dict(result)
