import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Import env
from dotenv import load_dotenv
def save_faiss():
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Load documents
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)

    # Create the vectorstore
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("./pages/faiss_index")

    print("FAISS index stored successfully.")

def save_graph():
    # Set up the Neo4j graph
    graph = Neo4jGraph()

    # Set up the LLM for graph transformation
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    llm_transformer = LLMGraphTransformer(llm=llm)

    # Load documents
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(pages)

    # Convert documents to graph documents and add to Neo4j
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    print("Documents saved to Neo4j graph successfully.")

# Call the functions
save_faiss()
save_graph()
