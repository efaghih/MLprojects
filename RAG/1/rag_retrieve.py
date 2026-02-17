import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def chunk_text(text: str, chunk_size: int = 200):
    # Super simple chunking by character count (good enough for a toy demo)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

def main():
    # 1) Load text
    path = os.path.join("data", "notes.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Chunk
    chunks = chunk_text(text, chunk_size=220)

    # 3) Embed locally
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # 4) Build FAISS index (cosine similarity via inner product on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # 5) Queries 
    queries = [
        "What is FAISS used for in RAG?",
        "Why can excess moisture cause odor issues?"
    ]

    for query in queries:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, ids = index.search(q_emb, k=2)

        print("\n" + "=" * 70)
        print("Query:", query)
        print("\nTop matches:")
        for rank, (i, score) in enumerate(zip(ids[0], scores[0]), start=1):
            print(f"\n#{rank}  score={float(score):.3f}")
            print(chunks[int(i)])

if __name__ == "__main__":
    main()