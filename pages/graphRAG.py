from __future__ import annotations

import streamlit as st
from langchain_core.runnables import RunnableLambda

from utils.config import get_config
from utils.graph import (
    format_neo4j_error,
    get_entity_chain,
    get_graph,
    structured_retriever,
)
from utils.rag_components import build_answer_chain
from utils.streamlit_helpers import render_evaluation_section, stop_if_missing_required_vars


CONFIG = get_config()

def graph_context(question: str) -> str:
    structured_data = structured_retriever(
        question=question,
        graph=get_graph(CONFIG),
        entity_chain=get_entity_chain(CONFIG),
        index_name=CONFIG.graph_fulltext_index,
        fulltext_limit=CONFIG.graph_fulltext_limit,
        relation_limit=CONFIG.graph_relation_limit,
    )
    return f"Structured graph context:\n{structured_data}"


@st.cache_resource(show_spinner=False)
def create_graph_chain():
    return build_answer_chain(
        context_runnable=RunnableLambda(graph_context),
        model=CONFIG.openai_graph_model,
        config=CONFIG,
    )


def main() -> None:
    st.title("GraphRAG Q&A and Evaluation")

    stop_if_missing_required_vars(CONFIG)

    try:
        graph_chain = create_graph_chain()
        get_graph(CONFIG)
    except Exception as exc:  # pragma: no cover - runtime connectivity errors
        st.error(f"Graph initialization failed:\n\n{format_neo4j_error(exc)}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GraphRAG Section")
        st.write("Ask a question using structured context from Neo4j.")
        user_question = st.text_input("Enter your question:", key="graph_question")

        if user_question:
            with st.spinner("Getting answer..."):
                try:
                    predicted_answer = graph_chain.invoke(user_question)
                    st.session_state["graph_predicted_answer"] = predicted_answer
                    st.write(f"**GraphRAG Predicted Answer:** {predicted_answer}")
                except Exception as exc:  # pragma: no cover - runtime connectivity errors
                    st.error(f"Graph query failed:\n\n{format_neo4j_error(exc)}")

    with col2:
        render_evaluation_section(
            user_question=user_question,
            actual_answer_key="graph_actual_answer",
            predicted_answer_key="graph_predicted_answer",
            config=CONFIG,
        )


main()
