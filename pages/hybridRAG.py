from __future__ import annotations

import streamlit as st
from langchain_core.runnables import RunnableLambda

from utils.config import get_config
from utils.graph import (
    get_entity_chain,
    get_graph,
    get_hybrid_vector_index,
    format_neo4j_error,
    structured_retriever,
)
from utils.rag_components import (
    build_answer_chain,
    format_documents,
)
from utils.streamlit_helpers import render_evaluation_section, stop_if_missing_required_vars


CONFIG = get_config()

def hybrid_context(question: str) -> str:
    structured_data = structured_retriever(
        question=question,
        graph=get_graph(CONFIG),
        entity_chain=get_entity_chain(CONFIG),
        index_name=CONFIG.graph_fulltext_index,
        fulltext_limit=CONFIG.graph_fulltext_limit,
        relation_limit=CONFIG.graph_relation_limit,
    )
    unstructured_documents = get_hybrid_vector_index(CONFIG).similarity_search(
        question,
        k=CONFIG.rag_top_k,
    )
    return (
        f"Structured graph context:\n{structured_data}\n\n"
        f"Unstructured vector context:\n{format_documents(unstructured_documents)}"
    )


@st.cache_resource(show_spinner=False)
def create_hybrid_chain():
    return build_answer_chain(
        context_runnable=RunnableLambda(hybrid_context),
        model=CONFIG.openai_graph_model,
        config=CONFIG,
    )


def main() -> None:
    st.title("HybridRAG Q&A and Evaluation")

    stop_if_missing_required_vars(CONFIG)

    try:
        hybrid_chain = create_hybrid_chain()
        get_graph(CONFIG)
        get_hybrid_vector_index(CONFIG)
    except Exception as exc:  # pragma: no cover - runtime connectivity errors
        st.error(f"Hybrid retriever initialization failed:\n\n{format_neo4j_error(exc)}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HybridRAG Section")
        st.write("Ask a question using graph + vector context from Neo4j.")
        user_question = st.text_input("Enter your question:", key="hybrid_question")

        if user_question:
            with st.spinner("Getting answer..."):
                try:
                    predicted_answer = hybrid_chain.invoke(user_question)
                    st.session_state["hybrid_predicted_answer"] = predicted_answer
                    st.write(f"**HybridRAG Predicted Answer:** {predicted_answer}")
                except Exception as exc:  # pragma: no cover - runtime connectivity errors
                    st.error(f"Hybrid query failed:\n\n{format_neo4j_error(exc)}")

    with col2:
        render_evaluation_section(
            user_question=user_question,
            actual_answer_key="hybrid_actual_answer",
            predicted_answer_key="hybrid_predicted_answer",
            config=CONFIG,
        )


main()
