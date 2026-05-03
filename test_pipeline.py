"""
test_pipeline.py - Validates chunking, embedding, and retrieval WITHOUT needing an API key.
Run: python test_pipeline.py
"""

import numpy as np
from embed import chunk_text, embed_texts, get_model
from retrieve import build_faiss_index, retrieve_top_k

print("=" * 60)
print("  Mini RAG Pipeline - Unit Test (No API key needed)")
print("=" * 60)

# ── Test 1: Chunking ──────────────────────────────────────────────────────────
print("\n[1] Testing chunk_text...")
sample = "This is a test document. " * 30   # ~750 chars
chunks = chunk_text(sample, chunk_size=200, overlap=40)
print(f"    Input length : {len(sample)} chars")
print(f"    Chunks created: {len(chunks)}")
print(f"    First chunk   : '{chunks[0][:60]}...'")
assert len(chunks) >= 2, "Expected at least 2 chunks"
print("    ✅ PASSED")

# ── Test 2: Embeddings ────────────────────────────────────────────────────────
print("\n[2] Testing embed_texts (downloads model on first run)...")
test_texts = [
    "Safety rules must be followed at all times.",
    "Project delays are caused by poor planning.",
    "Machine learning uses neural networks.",
]
embeddings = embed_texts(test_texts)
print(f"    Embedding shape: {embeddings.shape}")
assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"
print("    ✅ PASSED")

# ── Test 3: FAISS Index ───────────────────────────────────────────────────────
print("\n[3] Testing FAISS index build + retrieval...")
index = build_faiss_index(embeddings)
print(f"    Index size: {index.ntotal} vectors")

query = "What causes delays in projects?"
results = retrieve_top_k(query, index, test_texts, k=2)
print(f"    Query: '{query}'")
for r in results:
    print(f"    Rank {r['rank']}: '{r['chunk'][:60]}' (score: {r['score']:.4f})")

assert len(results) == 2
assert results[0]["rank"] == 1
print("    ✅ PASSED")

# ── Test 4: Full pipeline (no LLM) ────────────────────────────────────────────
print("\n[4] Testing RAGPipeline (retrieval only, no LLM)...")
from rag_pipeline import RAGPipeline
import os

pipeline = RAGPipeline(data_dir="data", chunk_size=300, top_k=3)
pipeline.build_index()
print(f"    Total chunks indexed: {len(pipeline.chunks)}")

# Manually test retrieval without calling LLM
from retrieve import retrieve_top_k as rtk
results = rtk("What causes project delays?", pipeline.index, pipeline.chunks, k=3)
print(f"\n    Query: 'What causes project delays?'")
for r in results:
    print(f"    Rank {r['rank']} (score {r['score']:.4f}): {r['chunk'][:80]}...")

print("    ✅ PASSED")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✅")
print("  Now set OPENROUTER_API_KEY and run: streamlit run app.py")
print("=" * 60)
