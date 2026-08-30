import streamlit as st
import faiss
import pickle
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline

PAGE_TITLE = "TravelAssist RAG"

st.set_page_config(page_title=PAGE_TITLE, layout="centered")

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return embedding_model, index, chunks, generator


embedding_model, index, chunks, generator = load_models()


def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]


def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a helpful travel support assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say:
"I will connect you to a human support agent."

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(prompt, max_length=256)
    return response[0]["generated_text"]


st.title("✨ TravelAssist AI")
st.write("Ask travel-related questions powered by a grounded RAG workflow.")

query = st.text_input("Ask your travel question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Generating a grounded answer..."):
            answer = generate_answer(query)
        st.success(answer)
