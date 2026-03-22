import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import torch

# -----------------------------
# Load FLAN-T5 BASE (Better Model)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF Chatbot (RAG - Interview Ready)")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector DB
        db = FAISS.from_documents(docs, embeddings)

    st.success("PDF processed successfully!")

    # -----------------------------
    # Query Section
    # -----------------------------
    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("Thinking..."):

            # Better retrieval
            results = db.similarity_search(query, k=3)

            if results:
                # Combine context
                context = " ".join([r.page_content for r in results])

                # Strong Prompt
                prompt = f"""
You are a professional AI assistant.

Answer the question clearly and completely using the context.
Give a meaningful explanation in 2-3 sentences.

Context:
{context}

Question:
{query}

Answer:
"""

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

                # Generate answer
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=120
                    )

                # Decode
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.subheader("📌 Answer:")
                st.success(answer)

                #  (Optional - show source)
                with st.expander("🔍 Source Context"):
                    for i, doc in enumerate(results):
                        st.write(f"Chunk {i+1}:")
                        st.write(doc.page_content[:300])
                        st.write("---")

            else:
                st.warning("No relevant answer found.")