# PDF_RAG_ChatBot
# 📄 PDF RAG Chatbot

An intelligent PDF Question Answering system built using **Retrieval-Augmented Generation (RAG)**.  
This application allows users to upload PDF documents and ask questions, generating context-aware answers using a local LLM.

## 🚀 Features
- 📂 Upload and process PDF documents  
- ✂️ Automatic text chunking  
- 🔍 Semantic search using FAISS  
- 🤖 Context-based answer generation  
- 📴 Works completely offline (no API required)  
- 💡 Simple and interactive UI using Streamlit
  
## 🧠 Tech Stack
- **Python**
- **Streamlit**
- **LangChain**
- **FAISS (Vector Database)**
- **HuggingFace Transformers (FLAN-T5)**
- **Sentence Transformers (Embeddings)**

## ⚙️ How It Works
1. 📄 PDF is uploaded and parsed  
2. ✂️ Text is split into smaller chunks  
3. 🔗 Chunks are converted into embeddings  
4. 🗄️ FAISS stores and retrieves relevant chunks  
5. 🤖 FLAN-T5 generates answers based on retrieved context  

## ▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py
