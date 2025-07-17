# LangChain RAG Streamlit App

A demo project for Retrieval-Augmented Generation (RAG) using LangChain, Hugging Face API, and Streamlit. The app lets you upload documents, ask questions, and get answers using a generative AI model with context retrieval.

## Features
- Upload and index documents (sample data included)
- Retrieve relevant context using ChromaDB
- Generate answers using a Hugging Face-hosted LLM
- Web UI with Streamlit
- Deployable to Hugging Face Spaces

## Setup
1. **Clone the repo**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Hugging Face API key**
   - Create a `.env` file in the project root:
     ```env
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
     ```
4. **Run locally**
   ```bash
   streamlit run app.py
   ```

## Deployment (Hugging Face Spaces)
1. Push this repo to your Hugging Face account as a new Space (Streamlit type).
2. Add your `HUGGINGFACEHUB_API_TOKEN` as a secret in the Space settings.
3. The app will build and deploy automatically.

---

This project is designed for technical interview prep and learning about RAG, LangChain, and prompt engineering.