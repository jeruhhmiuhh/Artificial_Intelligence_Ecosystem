"""
RAG_app.py
----------
Full Retrieval-Augmented Generation pipeline:
  1. Suppress noisy logs
  2. Load OpenAI credentials
  3. Define parameters
  4. Read pre-scraped document
  5. Split into chunks
  6. Embed with Sentence-Transformers & build FAISS index
  7. Retrieve relevant chunks
  8. Re-rank with a Cross-Encoder
  9. Answer questions via ChatGPT API
 10. Interactive Q&A loop
"""

# ---------------------------------------------------------------------------
# Step 3.1 – Suppress Noisy Logs
# ---------------------------------------------------------------------------
import logging
import warnings
import transformers.utils.logging as hf_logging

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Step 3.2 – ChatGPT API Credentials
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
import openai

load_dotenv()  # reads .env from the current directory
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Step 3.3 – Parameters
# ---------------------------------------------------------------------------
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

# Re-ranking parameters
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

# ---------------------------------------------------------------------------
# Step 3.4 – Read the Pre-scraped Document
# ---------------------------------------------------------------------------
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------------------------------------------------------------------
# Step 3.5 – Split into Appropriately-Sized Chunks
# ---------------------------------------------------------------------------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = splitter.split_text(text)

# ---------------------------------------------------------------------------
# Step 3.6 – Embed & Build FAISS Index
# ---------------------------------------------------------------------------
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(model_name)

print("Encoding document chunks — this may take a moment...")
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings, dtype=np.float32)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

print(f"FAISS index built with {faiss_index.ntotal} vectors (dim={dimension}).")

# ---------------------------------------------------------------------------
# Step 3.7 – Retrieval Function
# ---------------------------------------------------------------------------

def retrieve_chunks(question: str, k: int = top_k) -> list[str]:
    """
    Encodes `question` using the bi-encoder, searches the FAISS index for
    the top-k nearest neighbors, and returns the corresponding text chunks.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec, dtype=np.float32)
    _distances, I = faiss_index.search(q_arr, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# ---------------------------------------------------------------------------
# Step 3.8 – Cross-Encoder Re-Ranker
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """
    Returns a list with duplicate strings removed while preserving the
    first occurrence of each. Normalizes whitespace before comparing.
    """
    seen = set()
    result = []
    for item in items:
        normalized = " ".join(item.split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(item)
    return result


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    """
    Scores each (question, chunk) pair with the cross-encoder, sorts by
    score descending, and returns the top-m chunks after light deduplication.

    Note: does NOT re-encode with the bi-encoder — only the cross-encoder
    is used here for scoring.
    """
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)                      # higher = more relevant
    ranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _score, chunk in ranked[:m]]
    return dedupe_preserve_order(top_chunks)

# ---------------------------------------------------------------------------
# Step 3.9 – Q&A with ChatGPT
# ---------------------------------------------------------------------------

def answer_question(question: str) -> str:
    """
    Full RAG pipeline for a single question:
      1. Retrieve top_k candidate chunks via bi-encoder + FAISS.
      2. Re-rank with cross-encoder, keep top_m.
      3. Build a context string and call the ChatGPT API.
      4. Return the assistant's answer.
    """
    # Step 1: Coarse retrieval
    candidates = retrieve_chunks(question)

    # Step 2: Fine re-ranking
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)

    # Step 3: Build context
    context = "\n\n".join(relevant_chunks)

    # Step 4: Construct prompts
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the "
        "provided context. If the answer is not in the context, say you don't know."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 5: Call ChatGPT API
    client = openai.OpenAI(api_key=openai.api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",           # using gpt-4o (gpt-5 not yet publicly available)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Step 3.10 – Interactive Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("RAG system ready. Enter 'exit' or 'quit' to end.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
        print("Answer:", answer_question(question))
