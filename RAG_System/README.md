# 4.2 Retrieval-Augmented Generation – Reflection Report

## Selected Document

**Topic:** Tarot — History, Symbolism, and Esoteric Tradition  
**Source:** Wikipedia article on Tarot (https://en.wikipedia.org/wiki/Tarot) scraped via `text_extractor.py`  
**Output file:** `Selected_Document.txt`

The document covers the history of tarot cards, the Major and Minor Arcana, numerological and astrological associations, prominent tarot decks (Rider-Waite-Smith, Thoth), and modern spiritual practices. It was chosen because its rich symbolic vocabulary — suits, arcana, court cards, chakra associations — provides meaningful variety for testing retrieval quality across different question types.

---

## RAG Questions & Answers

The following three questions were asked of the RAG system using default parameters (`chunk_size=500`, `chunk_overlap=50`).

### Question 1: What are the four suits of the Minor Arcana and what elements do they represent?

**Answer:**  
The four suits of the Minor Arcana are Wands, Cups, Swords, and Pentacles (also called Coins or Disks). Wands correspond to Fire and deal with creativity, ambition, and career. Cups correspond to Water and govern emotions, relationships, and intuition. Swords correspond to Air and represent intellect, conflict, and communication. Pentacles correspond to Earth and cover material matters, finances, and health.

### Question 2: Which tarot card is associated with the zodiac sign Scorpio?

**Answer:**  
The Death card (card 13 of the Major Arcana) corresponds to the zodiac sign Scorpio. Despite its name, the Death card symbolizes endings, transitions, and transformation rather than literal death — a fitting association for Scorpio's themes of regeneration and deep change.

### Question 3: Who created the Rider-Waite-Smith tarot deck and what made it significant?

**Answer:**  
The Rider-Waite-Smith deck was published in 1909, designed by occultist Arthur Edward Waite and illustrated by Pamela Colman Smith, both members of the Hermetic Order of the Golden Dawn. Its significance lies in being the first widely distributed deck to feature fully illustrated scenic pip cards in the Minor Arcana, replacing the plain pip designs of earlier decks with narrative scenes — making intuitive reading far more accessible to the general public.

---

## Chunk Size & Overlap Experiments

| chunk_size | chunk_overlap | Observation |
|-----------|--------------|-------------|
| 500       | 50 (default) | Good balance. Answers were coherent and contextually complete. Retrieval returned focused, relevant passages. |
| 200       | 20           | Smaller chunks led to fragmented context. Answers sometimes missed connecting details (e.g., the element-suit relationship was split across chunks). |
| 1000      | 100          | Larger chunks captured more context per passage but retrieved fewer unique perspectives. Answers were longer but occasionally included off-topic information from the same large chunk. |

**Conclusion:** The default `chunk_size=500` with `chunk_overlap=50` worked best for this document. Smaller chunks hurt coherence; larger chunks diluted relevance. The overlap is critical for preserving sentences that straddle chunk boundaries.

---

## Five Deep-Dive Questions & AI Answers

### Q1: What does embedding dimensionality mean, and why does it matter for FAISS search?

**A:** Embedding dimensionality refers to the number of numerical values (dimensions) in the vector that represents a piece of text. The model `all-distilroberta-v1` produces 768-dimensional embeddings. Each dimension captures some aspect of semantic meaning learned during training. Dimensionality matters for FAISS because the index (`IndexFlatL2`) stores and searches vectors in this fixed-dimensional space using Euclidean (L2) distance — a lower distance means higher semantic similarity. If you change the embedding model, the dimension changes, and the FAISS index must be rebuilt from scratch with the correct dimension. Higher dimensionality can encode richer meaning but also increases memory use and search time.

### Q2: How does FAISS's IndexFlatL2 actually find nearest neighbors?

**A:** `IndexFlatL2` performs an exact brute-force search: it computes the L2 (Euclidean) distance between the query vector and *every* stored vector, then returns the `k` vectors with the smallest distances. This is guaranteed to be exact but becomes slow at very large scale (millions of vectors). For larger datasets, approximate FAISS indexes like `IndexIVFFlat` or `IndexHNSW` trade a small amount of accuracy for dramatically faster search by partitioning the space or building graph-based shortcuts.

### Q3: Why is chunk overlap important and what happens without it?

**A:** Chunk overlap ensures that sentences or ideas that fall near a chunk boundary appear in both the preceding and following chunks. Without overlap, a key sentence that happens to sit at the boundary between two 500-character chunks might end up in neither chunk in a useful form — cut off mid-thought. For example, if a sentence about "the High Priestess corresponds to the Moon" falls at the very end of one chunk, a question about the Moon's tarot association might miss it entirely without overlap. Overlap at 50 characters gives a small but meaningful buffer that preserves cross-boundary context.

### Q4: What is the difference between a bi-encoder and a cross-encoder, and why do we use both in this pipeline?

**A:** A **bi-encoder** (like `all-distilroberta-v1`) independently encodes the query and each document chunk into vectors, then compares them via cosine or L2 distance. This is fast because all chunk vectors are pre-computed once and stored in FAISS — retrieval is a single matrix search. However, bi-encoders are less accurate because the query and document never "see" each other during encoding.

A **cross-encoder** (like `ms-marco-MiniLM-L-6-v2`) takes a (query, chunk) pair as a single input and runs them through the model together, producing a relevance score. This is much more accurate because the model can attend to interactions between the query and document tokens. The trade-off is speed — it cannot be pre-computed and must run for every candidate pair at query time.

This pipeline uses both in a two-stage approach: the fast bi-encoder retrieves 20 candidates, then the accurate cross-encoder re-ranks them to select the best 8. This balances speed and precision.

### Q5: How does the system prompt affect the quality of ChatGPT's answers?

**A:** The system prompt frames the model's role and constrains its behavior before the user message is even seen. In this RAG pipeline, the system prompt instructs the model to act as a "knowledgeable assistant that answers questions based on the provided context" and to say "I don't know" if the answer is not in the context. This is critical for two reasons: (1) it prevents hallucination by anchoring the model to the retrieved context rather than its parametric (training-time) knowledge, and (2) it sets a conservative fallback so the model doesn't invent plausible-sounding but incorrect answers when the retrieval fails. Changing the system prompt — for instance, removing the context constraint — would cause the model to rely on its general training knowledge, undermining the purpose of RAG.

---

## How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to .env
echo "OPENAI_API_KEY=your-key-here" > .env

# 4. Scrape the document
python text_extractor.py

# 5. Run the RAG system
python RAG_app.py
```
