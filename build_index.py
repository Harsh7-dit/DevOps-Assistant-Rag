import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from load_documents import load_documents

print("Loading documents...")
documents = load_documents()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["chunk"] for doc in documents]

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]

print("Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.bin")

with open("metadata.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index and metadata saved successfully.")
