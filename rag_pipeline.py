import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# --------------------------------------------------
# Load Models
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading FLAN-T5 model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

print("Models loaded successfully.\n")


# --------------------------------------------------
# Load FAISS Vector Store
# --------------------------------------------------

print("Loading FAISS vector store...")

index = faiss.read_index(
    "faiss_index.index"
)

with open("chunks.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} travel documents.\n")


# --------------------------------------------------
# Retrieve Relevant Travel Documents
# --------------------------------------------------

def retrieve(query, top_k=3):
    """
    Convert the user query into an embedding and retrieve
    the most semantically similar travel documents.
    """

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        results.append({
            "document": documents[index_id],
            "distance": float(distance)
        })

    return results


# --------------------------------------------------
# Generate Grounded Answer
# --------------------------------------------------

def generate_answer(query):

    # Retrieve top 3 relevant documents
    results = retrieve(query, top_k=3)

    # --------------------------------------------------
    # Display Retrieved Documents
    # --------------------------------------------------

    print("\n--- RETRIEVED DOCUMENTS ---")

    for i, result in enumerate(results, 1):

        print(f"\nDocument {i}")
        print("Distance:", result["distance"])
        print(result["document"][:400])

    print("\n--------------------------\n")

    # --------------------------------------------------
    # Build SHORT Context
    # --------------------------------------------------

    context_parts = []

    for i, result in enumerate(results, 1):

        document = result["document"]

        # Keep only a short portion of each document
        document = document[:350]

        context_parts.append(
            f"Document {i}:\n{document}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------
    # Short Prompt
    # --------------------------------------------------

    prompt = f"""
You are a travel recommendation assistant.

Use only the context below.

Choose the destination that best matches the
user's preferences.

Explain your choice briefly.

Do not invent information.

Context:
{context}

Question:
{query}

Answer:
"""

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    response = generator(
        prompt,
        max_length=100,
        do_sample=False
    )

    return response[0]["generated_text"]

# --------------------------------------------------
# CLI Application
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("              TravelAssist AI")
    print("          RAG Travel Assistant")
    print("=" * 70)

    print("\nType 'exit' to stop.\n")

    while True:

        query = input("Ask a travel question: ")

        if query.lower().strip() == "exit":
            print("\nGoodbye!")
            break

        if not query.strip():
            print("Please enter a question.\n")
            continue

        answer = generate_answer(query)

        print("\nAnswer:")
        print(answer)

        print("\n" + "=" * 70 + "\n")