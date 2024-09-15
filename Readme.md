# RAG-based PDF Question Answering with Accuracy Evaluation

This project allows you to ask questions over your PDF files using **Vector RAG**, **Graph RAG**, and **Hybrid RAG**. It evaluates the accuracy of responses from each pipeline by letting you enter the actual answer. An LLM (GPT-4) acts as a judge, comparing the RAG system’s answer against the actual answer you provide and the question.

- **Vector RAG**: Uses FAISS as the vector store.
- **Graph RAG**: Uses Neo4j for graph storage.
- **Hybrid RAG**: Combines both FAISS and Neo4j.

You can upload multiple custom PDF files. Place your files in the `data` folder, and run the commands below.

The project uses a **Streamlit** interface with three separate pages for **Vector RAG**, **Graph RAG**, and **Hybrid RAG**.

HELPFUL LINK: https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/

## Environment Setup

1. Create a Neo4j instance, and add your username, password, and instance name to the `.env` file.
2. Create an OpenAI API key, and add it to the `.env` file.

## File Structure

- **`data/`**: This folder contains all the PDF files. Add your own PDFs here.
- **`updateDB.py`**: Ingests files from the `data/` folder into the databases. 
    - Updates the **Neo4j** database remotely.
    - Updates the **FAISS** vector database locally inside the `faiss_index` folder.
- **`app.py`**: The entry point for the **Streamlit** app.
    - **`pages/`**: Contains three pages for the three types of RAG (Vector RAG, Graph RAG, and Hybrid RAG). Each page has code relevant to that specific pipeline.

## How to Run

1. First make sure you have created a virtual environment in Python. To create it, run the command: `python -m venv myenv` and then activate it using `myenv/Scripts/activate`
2. Install requirements using `pip install -r requirements.txt`
3. Then just run `streamlit run app.py` to launch the interface.


















