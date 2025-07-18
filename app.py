import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import requests

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

# --- RAG Retrieval and Answer Generation ---
def retrieve_context(question, vectordb, k=3):
    # Embed the question and retrieve top-k relevant chunks
    docs_and_scores = vectordb.similarity_search_with_score(question, k=k)
    # Filter by a reasonable similarity threshold (optional)
    threshold = 0.7  # Lower is more similar; adjust as needed
    relevant_chunks = [doc.page_content for doc, score in docs_and_scores if score < threshold]
    return relevant_chunks

# Hugging Face LLM API call
def call_hf_llm(prompt, model="HuggingFaceH4/zephyr-7b-beta"):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        result = response.json()
        # Handle both string and list outputs
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    else:
        return f"[Error] Hugging Face API: {response.status_code} - {response.text}"

# Main chat interface
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            relevant_chunks = retrieve_context(user_question, vectordb, k=3)
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}\n\nAnswer:"
                st.info("Relevant context retrieved and sent to LLM.")
            else:
                prompt = f"Question:\n{user_question}\n\nAnswer:"
                st.info("No relevant context found. Falling back to LLM-only mode.")
            answer = call_hf_llm(prompt)
            st.write(f"**Answer:** {answer}")
            if relevant_chunks:
                with st.expander("Show retrieved context"):
                    st.write(context)

st.markdown("---")
st.markdown("_RAG retrieval and answer generation are now implemented!_\n\nYou can upload your own documents or use the sample data to ask questions.")