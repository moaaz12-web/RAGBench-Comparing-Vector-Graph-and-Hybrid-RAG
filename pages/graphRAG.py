import os
import streamlit as st
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Import env
from dotenv import load_dotenv
load_dotenv()

# Initialize Neo4j and OpenAI models
graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# Create Fulltext index in the Neo4j database
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="Any scientific, technological terms and concepts that appear in the text, along with their properties and attirbutes, and their working mechanism and theory",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting scientific and technical terminologies and concepts from the text, along with its working mechanism and background.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# print(entity_chain.invoke({"question":"What is the attention mechanism in transformers?"}))
#! Outputs: ['attention mechanism', 'transformers']

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question}) # Extract all entities from user question using LLM
    
    for entity in entities.names: # entities.name contains a list of all entities identified in user question by the LLM
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:10})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    final_data = f"""Structured data:
{structured_data}"""
    return final_data


# Define the entity extraction chain and question answering chain (GraphRAG)
def create_graph_chain():
    template="""You are a chatbot that answers questions from the context below. You need to be accurate in your responses. 
    Provide descriptive answers based on the provided context. DO NOT MAKE STUFF UP BY YOURSELF.
    Context : {context}
    Question : {question}
    Response : 
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": retriever,  # This function collects structured data from the graph
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# Function to evaluate predicted answer
def evaluate_predicted_answer(question, actual_answer, predicted_answer):
    # Initialize LLM with gpt-4 model
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
    st.title("GraphRAG Q&A and Evaluation")

    # Create two columns for GraphRAG and Evaluation side by side
    col1, col2 = st.columns(2)

    # Left Side: GraphRAG Section
    with col1:
        st.subheader("GraphRAG Section")
        st.write("Ask a question based on structured data from the graph database.")

        # Create the GraphRAG chain
        graph_chain = create_graph_chain()

        # Input for user question and actual answer
        user_question = st.text_input("Enter your question:")

        # Section to display the predicted answer
        if user_question:
            with st.spinner("Getting answer..."):
                predicted_answer = graph_chain.invoke(user_question)
                st.write(f"**GraphRAG Predicted Answer:** {predicted_answer}")

    # Right Side: Evaluation Section
    with col2:
        st.subheader("Evaluation Section")
        st.write("This section evaluates how close the predicted answer is to the actual answer.")
        actual_answer = st.text_input("Enter the actual answer:")

        # Check if both question and actual answer are provided
        if user_question and actual_answer:

            # Create and display JSON output for evaluation
            evaluation_json = evaluate_predicted_answer(user_question, actual_answer, predicted_answer)
            st.subheader("LLM Checker Output")
            st.json(evaluation_json)

        # Inform the user to provide both fields for evaluation
        elif user_question and not actual_answer:
            st.info("Provide an actual answer for evaluation.")


    # graph.close()
# Run the Streamlit app
main()
