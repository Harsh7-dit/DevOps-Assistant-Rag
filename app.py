from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load Everything Once ----------
print("Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")

with open("metadata.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FLAN-T5-small...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# ---------- Request Schema ----------
class Question(BaseModel):
    question: str

# ---------- Retrieval ----------
def retrieve(query, top_k=5):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return [documents[i]["chunk"] for i in indices[0]]

# ---------- API Endpoint ----------
@app.post("/ask")
def ask_question(data: Question):

    if data.question.lower() in ["exit", "quit"]:
        os._exit(0)
    context = retrieve(data.question)
    combined_context = "\n\n".join(context)

    prompt = f"""
You are a helpful DevOps assistant.

Use only the context below to answer.
Provide a detailed explanation.

Context:
{combined_context}

Question:
{data.question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
    )

    return {
        "answer": result[0]["generated_text"]
    }
