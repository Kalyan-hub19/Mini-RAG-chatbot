"""
embed.py - Handles text chunking and embedding generation
Uses sentence-transformers with all-MiniLM-L6-v2 model
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once (reused across calls)
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _model


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Raw text to chunk
        chunk_size: Max characters per chunk
        overlap: Characters of overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed
    
    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype='float32')
