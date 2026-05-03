"""
retrieve.py - FAISS-based vector search for relevant chunks
"""

import faiss
import numpy as np
from embed import embed_texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat L2 index from embeddings.
    
    Args:
        embeddings: numpy array of shape (n, dim)
    
    Returns:
        FAISS index with all embeddings added
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_top_k(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    k: int = 3
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    
    Args:
        query: User's question string
        index: FAISS index containing chunk embeddings
        chunks: List of original text chunks (parallel to index)
        k: Number of chunks to retrieve
    
    Returns:
        List of dicts with 'chunk', 'score', and 'rank'
    """
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(chunks):
            results.append({
                "rank": rank + 1,
                "chunk": chunks[idx],
                "score": float(dist),       # lower L2 = more similar
                "index": int(idx)
            })
    return results
