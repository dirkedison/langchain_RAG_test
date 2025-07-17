import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="LangChain RAG Demo", layout="wide")
st.title("LangChain RAG Demo (Hugging Face + Streamlit)")

# Sidebar for document upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text files", type=["txt"], accept_multiple_files=True
)

# --- Document Ingestion, Chunking, Embedding, and Vector Store Setup ---
def load_documents(files):
    docs = []
    for file in files:
        # Save uploaded file to a temp file for LangChain loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        loader = TextLoader(tmp_path, encoding="utf-8")
        docs.extend(loader.load())
        os.remove(tmp_path)
    return docs

# Sample data if no files uploaded
SAMPLE_TEXT = """
LangChain is a framework for developing applications powered by language models. It enables applications such as chatbots, Generative Question-Answering (GQA), summarization, and more.

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with generative models to provide more accurate and up-to-date answers by grounding responses in external data sources.
"""

def get_documents():
    if uploaded_files:
        return load_documents(uploaded_files)
    else:
        # Use sample data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(SAMPLE_TEXT)
            tmp_path = tmp.name
        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        os.remove(tmp_path)
        return docs

# Split documents into chunks
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# Embed and store in ChromaDB
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=".chromadb")
    return vectordb

# Only run ingestion/embedding once per session (cache)
@st.cache_resource(show_spinner=True)
def setup_vector_store():
    docs = get_documents()
    chunks = chunk_documents(docs)
    vectordb = embed_and_store(chunks)
    return vectordb

vectordb = setup_vector_store()

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded and indexed.")
else:
    st.sidebar.info("No files uploaded. Using sample data.")

# Main chat interface
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        st.info("Context retrieved and ready for RAG pipeline (answer generation not yet implemented)...")
        st.write("**Answer:** _This is a placeholder. The real answer will appear here._")

st.markdown("---")
st.markdown("_Document ingestion, chunking, and embedding are now implemented. Next: retrieval and answer generation._")