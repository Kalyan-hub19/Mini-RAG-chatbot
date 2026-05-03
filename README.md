
# 🔍 Mini RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG)** system built with FAISS, Sentence Transformers, and Mistral via OpenRouter.



## 🚀 Live Demo
```
> Deploy link here (Streamlit Cloud / HuggingFace Spaces)

## 🧠 How It Works
```
User Query
    │
    ▼
Embed Query (all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Search → Top-K Chunks
    │
    ▼
Build Grounded Prompt (context + query)
    │
    ▼
LLM (Mistral-7B via OpenRouter)
    │
    ▼
Grounded Answer + Retrieved Chunks shown in UI

```


## 📂 Project Structure

```
mini-rag/
│
├── data/
│   ├── doc1.txt          # Project management docs
│   ├── doc2.txt          # Workplace safety
│   └── doc3.txt          # AI & ML basics
│
├── app.py                # Streamlit UI
├── rag_pipeline.py       # Orchestrates full pipeline
├── embed.py              # Chunking + embedding
├── retrieve.py           # FAISS search
├── generate.py           # LLM answer generation
├── test_pipeline.py      # Unit tests (no API key needed)
│
├── requirements.txt
└── README.md

```

## ⚙️ Setup & Run

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd mini-rag
pip install -r requirements.txt
```

### 2. Get a free OpenRouter API key
- Visit [https://openrouter.ai](https://openrouter.ai)
- Create an account and get a free API key
- Mistral-7B is free to use

### 3. Set your API key
```bash
export OPENROUTER_API_KEY="your-key-here"
```

Or enter it directly in the Streamlit sidebar at runtime.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Test without API key (retrieval only)
```bash
python test_pipeline.py
```

---

## 🏗️ Design Choices

### Embedding Model: `all-MiniLM-L6-v2`
- Lightweight (22M params, 80MB) — fast even on CPU
- 384-dimensional embeddings — good semantic quality
- Best balance of speed vs. accuracy for small RAG systems

### Chunk Size: 300 chars, 50 overlap
- 300 chars ≈ 2–3 sentences — enough semantic meaning per chunk
- 50-char overlap prevents answers from being cut at chunk boundaries
- Tested: smaller chunks (100) hurt retrieval quality; larger (600) dilutes similarity

### Vector Store: FAISS (IndexFlatL2)
- Exact nearest-neighbor search — no approximation errors
- Zero setup — runs in-memory, no external database
- IndexFlatL2 is ideal for small corpora (<10k chunks)
- For scale: switch to `IndexIVFFlat` with training

### LLM: Mistral-7B via OpenRouter
- Free tier available on OpenRouter
- Strong instruction following
- OpenAI-compatible API → easy to swap models

### Grounding Strategy
The prompt explicitly instructs the model to:
1. Answer **only** from the provided context chunks
2. Say "I don't know" if the answer isn't in the context
3. Use `temperature=0.1` for factual, low-creativity responses

```python
prompt = """
You are a helpful assistant. Answer ONLY using the context below.
If the answer cannot be found, say "I don't know based on the provided documents."
"""



## 🧪 Test Questions & Evaluation

| Question | Expected Retrieval | Hallucination? |
|---|---|---|
| "What causes project delays?" | doc1 (project management) | None — grounded |
| "What safety rules must workers follow?" | doc2 (safety) | None |
| "What is Retrieval-Augmented Generation?" | doc3 (AI/ML) | None |
| "Who is the CEO of Apple?" | None relevant | Returns "I don't know" ✅ |

### Findings
- **Retrieval accuracy**: High for domain-specific queries matching document vocabulary
- **Hallucination**: Zero when context is present; correctly declines off-topic questions
- **Edge case**: Paraphrased questions (e.g., "schedule overruns" for "project delays") — FAISS still retrieves correctly because sentence embeddings capture semantic similarity, not just keywords

---

## 🚀 Deployment (Streamlit Cloud)

1. Push code to GitHub (ensure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` as entry point
4. Add `OPENROUTER_API_KEY` in the Secrets section
5. Deploy ✅

---

## 🔮 Possible Extensions

- [ ] Support PDF/DOCX ingestion
- [ ] Persist FAISS index to disk
- [ ] Add reranking (cross-encoder) for better retrieval
- [ ] Switch to `IndexIVFFlat` for large document sets
- [ ] Add chunk-level citation highlighting in UI
- [ ] Evaluation with RAGAS framework

# Mini-RAG-chatbot
