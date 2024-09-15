import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Import env
from dotenv import load_dotenv
load_dotenv()

# Define the prompt template
prompt_str = """You are a chatbot that answers questions from the context below. You need to be accurate in your responses. 
Provide descriptive answers based on the provided context. DO NOT MAKE STUFF UP BY YOURSELF.
Context : {contexte}
Question : {question}
Response : """


# Define the prompt and output parser
prompt = ChatPromptTemplate.from_template(prompt_str)
output_parser = StrOutputParser()

# Create the chain for SimpleRAG
def create_chain():
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        os.path.join(os.path.dirname(__file__), "faiss_index"), embeddings, allow_dangerous_deserialization=True
    )

    # Define the retrievers
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    # Create the retrieval chain
    retrieval = RunnableParallel(
        {"contexte": retriever, "question": RunnablePassthrough()}
    )
    chain = retrieval | prompt | llm | output_parser

    return chain


# Function to create JSON output for evaluation
# Function to evaluate predicted answer
def evaluate_predicted_answer(question, actual_answer, predicted_answer):
    # Initialize LLM with GPT-4 model
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    # Define the prompt template
    prompt_str = """You are a critical evaluator. You will be given a question, an actual (true) answer, and a predicted answer. You need to rate the predicted answer on a scale of 0 to 10 based on how closely it resembles the actual answer contextually. If the predicted answer is contextually diffrerent than the actual answer, you give a babd rating. Be highly critical when giving rating. You need to output in JSON format with the rating and your justification. So the output should strictly be a rating and a justification in key value pair, as in JSON format. Do not write anything else, do not provide any other message other than the JSON format.
    Below are the question, the actual (true) answer, and the predicted answer:
    {question}
    {actual_answer}
    {predicted_answer}
    """

    prompt = ChatPromptTemplate.from_template(prompt_str)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    res = chain.invoke({
        "question": question,
        "actual_answer": actual_answer,
        "predicted_answer": predicted_answer
    })

    return res


def main():
    st.title("SimpleRAG PDF Question-Answering App")

    # Create two columns for SimpleRAG and Evaluation side by side
    col1, col2 = st.columns(2)

    # Left Side: SimpleRAG
    with col1:
        st.subheader("SimpleRAG Section")
        st.write("Ask a question based on the pre-loaded PDF document.")

        # Create the SimpleRAG chain
        chain = create_chain()

        # Input for user question and actual answer
        user_question = st.text_input("Enter your question:")

        # Section to display the predicted answer
        if user_question:
            with st.spinner("Getting answer..."):
                predicted_answer = chain.invoke(user_question)
                st.write(f"**SimpleRAG Predicted Answer:** {predicted_answer}")

    # Right Side: Evaluation
    with col2:
        st.subheader("Evaluation Section")
        st.write("This section evaluates how close the predicted answer is to the actual answer.")
        actual_answer = st.text_input("Enter the actual answer:")


        # Check if both question and actual answer are provided
        if user_question and actual_answer:
            
            # Create and display JSON output for evaluation
            evaluation_json =  evaluate_predicted_answer(user_question, actual_answer, predicted_answer)
            st.subheader("LLM checker output")
            st.json(evaluation_json)

        # Inform the user to provide both fields for evaluation
        elif user_question and not actual_answer:
            st.info("Provide an actual answer for evaluation.")

# Run the Streamlit app
main()
