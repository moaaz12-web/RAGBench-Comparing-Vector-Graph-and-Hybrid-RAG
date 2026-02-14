from __future__ import annotations

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda

from utils.config import get_config
from utils.rag_components import build_answer_chain, build_embeddings, format_documents
from utils.streamlit_helpers import (
    render_evaluation_section,
    stop_if_missing_required_vars,
)


CONFIG = get_config()


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> FAISS:
    if not CONFIG.faiss_index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found at '{CONFIG.faiss_index_dir}'. Run updateDB.py first."
        )
    return FAISS.load_local(
        str(CONFIG.faiss_index_dir),
        build_embeddings(CONFIG),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def create_chain():
    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG.rag_top_k},
    )
    return build_answer_chain(
        context_runnable=retriever | RunnableLambda(format_documents),
        model=CONFIG.openai_chat_model,
        config=CONFIG,
    )


def main() -> None:
    st.title("SimpleRAG PDF Question-Answering App")

    stop_if_missing_required_vars(CONFIG)

    try:
        chain = create_chain()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SimpleRAG Section")
        st.write("Ask a question based on the indexed PDF documents.")
        user_question = st.text_input("Enter your question:", key="simple_question")

        if user_question:
            with st.spinner("Getting answer..."):
                predicted_answer = chain.invoke(user_question)
                st.session_state["simple_predicted_answer"] = predicted_answer
                st.write(f"**SimpleRAG Predicted Answer:** {predicted_answer}")

    with col2:
        render_evaluation_section(
            user_question=user_question,
            actual_answer_key="simple_actual_answer",
            predicted_answer_key="simple_predicted_answer",
            config=CONFIG,
        )


main()
