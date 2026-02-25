import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")

with open("metadata.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FLAN-T5-small...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def retrieve(query, top_k=3):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return [documents[i]["chunk"] for i in indices[0]]

def ask(query):
    context = retrieve(query, top_k=5)
    combined_context = "\n\n".join(context)

    prompt = f"""
You are a helpful DevOps assistant.

Use ONLY the provided context to answer the question.
Provide a detailed explanation.
Explain concepts clearly.
Use step-by-step reasoning if appropriate.

Context:
{combined_context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
    )

    return result[0]["generated_text"]
if __name__ == "__main__":
    print("DevOps RAG Assistant Ready!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Ask: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            break

        if not question:
            continue

        answer = ask(question)
        print("\nAnswer:\n", answer)
