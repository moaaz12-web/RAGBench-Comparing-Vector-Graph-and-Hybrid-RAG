from __future__ import annotations

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_config
from utils.graph import create_neo4j_graph
from utils.rag_components import build_chat_model, build_embeddings


CONFIG = get_config()


def load_pdfs():
    loader = DirectoryLoader(
        str(CONFIG.data_dir),
        glob=CONFIG.pdf_glob,
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No PDF documents found in '{CONFIG.data_dir}' using glob '{CONFIG.pdf_glob}'."
        )
    return documents


def save_faiss() -> None:
    documents = load_pdfs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.faiss_chunk_size,
        chunk_overlap=CONFIG.faiss_chunk_overlap,
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, build_embeddings(CONFIG))
    CONFIG.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(CONFIG.faiss_index_dir))
    print(f"FAISS index stored successfully at '{CONFIG.faiss_index_dir}'.")


def save_graph() -> None:
    graph = create_neo4j_graph(CONFIG)
    llm = build_chat_model(
        model=CONFIG.openai_ingest_model,
        temperature=0.0,
        config=CONFIG,
    )
    llm_transformer = LLMGraphTransformer(llm=llm)
    documents = load_pdfs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.graph_chunk_size,
        chunk_overlap=CONFIG.graph_chunk_overlap,
    )
    graph_documents = llm_transformer.convert_to_graph_documents(
        text_splitter.split_documents(documents)
    )
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )
    print("Documents saved to Neo4j graph successfully.")


def main() -> None:
    if CONFIG.missing_required_vars:
        raise EnvironmentError(
            "Missing required environment variable(s): "
            + ", ".join(CONFIG.missing_required_vars)
        )
    save_faiss()
    save_graph()


if __name__ == "__main__":
    main()
