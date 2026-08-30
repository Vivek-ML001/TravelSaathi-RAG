from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from data_processing import (
    load_travel_dataset,
    create_travel_documents
)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(documents):
    print(f"Generating embeddings for {len(documents)} documents...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def save_vector_store(index, documents):

    faiss.write_index(
        index,
        "faiss_index.index"
    )

    with open("chunks.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("\nVector store saved successfully.")
    print(f"Documents stored: {len(documents)}")


if __name__ == "__main__":

    # Load travel dataset
    df = load_travel_dataset(
        "data/wanderlust_destinations.csv"
    )

    # Convert rows into RAG documents
    documents = create_travel_documents(df)

    # Create FAISS index
    index = create_vector_store(documents)

    # Save FAISS + documents
    save_vector_store(
        index,
        documents
    )