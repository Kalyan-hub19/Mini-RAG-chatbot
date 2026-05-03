"""
rag_pipeline.py - Orchestrates the full RAG pipeline:
  1. Load & chunk documents
  2. Embed chunks
  3. Build FAISS index
  4. On query: retrieve → generate → return result
"""

import os
from embed import chunk_text, embed_texts, get_model
from retrieve import build_faiss_index, retrieve_top_k
from generate import generate_answer


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, data_dir: str = "data", chunk_size: int = 300, overlap: int = 50, top_k: int = 5, api_key: str = ""):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.api_key = api_key

        self.chunks: list[str] = []        # All text chunks
        self.chunk_sources: list[str] = [] # Source filename per chunk
        self.index = None                  # FAISS index

    # ── Indexing ────────────────────────────────────────────────────────────

    def load_documents(self) -> dict[str, str]:
        """Read all .txt files from data_dir."""
        docs = {}
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")

        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith(".txt"):
                path = os.path.join(self.data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    docs[filename] = f.read()

        if not docs:
            raise ValueError(f"No .txt files found in '{self.data_dir}'.")
        return docs

    def build_index(self):
        """Load docs → chunk → embed → build FAISS index."""
        print("📂 Loading documents...")
        docs = self.load_documents()
        print(f"   Loaded {len(docs)} document(s): {list(docs.keys())}")

        print("✂️  Chunking documents...")
        for filename, text in docs.items():
            doc_chunks = chunk_text(text, self.chunk_size, self.overlap)
            self.chunks.extend(doc_chunks)
            self.chunk_sources.extend([filename] * len(doc_chunks))
        print(f"   Created {len(self.chunks)} chunks total.")

        print("🔢 Generating embeddings...")
        # Warm up the model
        get_model()
        embeddings = embed_texts(self.chunks)
        print(f"   Embeddings shape: {embeddings.shape}")

        print("🗂️  Building FAISS index...")
        self.index = build_faiss_index(embeddings)
        print("✅ Index ready!\n")

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Run a full RAG query.

        Returns:
            {
                "question": str,
                "retrieved": [{"rank", "chunk", "score", "source"}],
                "answer": str
            }
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Step 1: Retrieve
        results = retrieve_top_k(question, self.index, self.chunks, self.top_k)

        # Attach source filenames
        for r in results:
            r["source"] = self.chunk_sources[r["index"]]

        # Step 2: Generate
        retrieved_texts = [r["chunk"] for r in results]
        answer = generate_answer(retrieved_texts, question, self.api_key)

        return {
            "question": question,
            "retrieved": results,
            "answer": answer,
        }
