# Simple RAG Practice Project (Windows + VS Code)

This is a very small project to practice the core idea of **RAG (Retrieval Augmented Generation)** using:
- **Local embeddings** (SentenceTransformers)
- **Vector search** (FAISS)
- An optional **LLM generation step** (OpenAI API)

The project is intentionally simple so you can understand each moving part.

---

## Goal

Build a minimal RAG pipeline that can answer a question using your own notes as a source.

The pipeline is:

1. Put text in `data/notes.txt`
2. Split the text into small chunks
3. Convert each chunk into an embedding vector (numbers)
4. Store vectors in a FAISS index
5. For a user query, embed the query
6. Use FAISS to retrieve the most similar chunks (top-k)
7. (Optional) Send the retrieved chunks as context to an LLM to generate a final answer


