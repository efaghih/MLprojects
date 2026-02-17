import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def chunk_text(text: str, chunk_size: int = 220):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

def build_index(chunks, model):
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    return index, emb

def retrieve(query, chunks, model, index, k=3):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, k=k)
    results = []
    for i, s in zip(ids[0], scores[0]):
        results.append((chunks[int(i)], float(s)))
    return results

def main():
    load_dotenv()
    client = OpenAI()

    # Load notes
    with open(os.path.join("data", "notes.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=220)

    # Local embeddings + FAISS
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_index(chunks, emb_model)

    # Ask
    query = "Why can excess moisture cause odor issues?"

    # Retrieve top context
    top = retrieve(query, chunks, emb_model, index, k=3)
    context = "\n\n".join([f"- {t[0]}" for t in top])

    # Generate answer using retrieved context
    prompt = f"""You are answering using ONLY the context below.
If the context is not enough, say: "I don't have enough information in the notes."

Context:
{context}

Question: {query}
Answer (2-5 sentences):"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    print("Query:", query)
    print("\nRetrieved context:")
    for i, (txt, score) in enumerate(top, 1):
        print(f"\n#{i} score={score:.3f}\n{txt}")

    print("\n" + "=" * 70)
    print("LLM answer:\n")
    print(resp.output_text)

if __name__ == "__main__":
    main()