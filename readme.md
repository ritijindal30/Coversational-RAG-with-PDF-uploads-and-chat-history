# Conversational RAG with PDF Uploads and Chat History

This project is a **Conversational RAG (Retrieval-Augmented Generation) application** built with **Streamlit** and **LangChain**.  
It allows you to upload PDFs, extract their content, and **chat with the documents interactively**, while maintaining **chat history** across sessions.  
The app uses **Groq’s LLMs** for answering questions and **Chroma** as the vector store for document retrieval.

---

## 🚀 Features
- 📄 Upload one or multiple **PDF files**.
- 🧠 Chunk and embed documents using **HuggingFace sentence-transformers**.
- 🔎 Store embeddings in **Chroma** and retrieve context on demand.
- 💬 Ask questions based on document content.
- 📝 Maintains **chat history** per session.
- ⚡ Powered by **Groq API** for LLM inference.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – RAG pipeline
- [Chroma](https://www.trychroma.com/) – Vector database
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) – Text embeddings
- [Groq LLM](https://groq.com/) – Fast inference API
- [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pdf) – PDF parsing

---

