from __future__ import annotations

import streamlit as st

from .config import AppConfig
from .rag_components import evaluate_answer


def stop_if_missing_required_vars(config: AppConfig) -> None:
    if not config.missing_required_vars:
        return
    st.error(
        "Missing required environment variable(s): "
        + ", ".join(config.missing_required_vars)
    )
    st.stop()


def render_evaluation_section(
    *,
    user_question: str,
    actual_answer_key: str,
    predicted_answer_key: str,
    config: AppConfig,
) -> None:
    st.subheader("Evaluation Section")
    st.write("Evaluate how close the predicted answer is to the actual answer.")
    actual_answer = st.text_input("Enter the actual answer:", key=actual_answer_key)

    if user_question and actual_answer:
        predicted_answer = st.session_state.get(predicted_answer_key, "")
        if not predicted_answer:
            st.warning("Generate a predicted answer first.")
            return
        evaluation_json = evaluate_answer(
            question=user_question,
            actual_answer=actual_answer,
            predicted_answer=predicted_answer,
            config=config,
        )
        st.subheader("LLM Checker Output")
        st.json(evaluation_json)
    elif user_question and not actual_answer:
        st.info("Provide an actual answer for evaluation.")
