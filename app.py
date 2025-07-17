 import streamlit as st
import os
from dotenv import load_dotenv

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

# Placeholder for document ingestion and indexing
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded. (Ingestion not yet implemented)")
else:
    st.sidebar.info("No files uploaded. Using sample data.")

# Main chat interface
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        # Placeholder for RAG pipeline
        st.info("Retrieving context and generating answer (not yet implemented)...")
        st.write("**Answer:** _This is a placeholder. The real answer will appear here._")

st.markdown("---")
st.markdown("_This is a starter template. Backend RAG pipeline and document ingestion will be implemented next._")