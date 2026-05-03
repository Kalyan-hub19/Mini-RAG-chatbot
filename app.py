"""
app.py - Streamlit chatbot UI for the Mini RAG system
Run with: streamlit run app.py
"""

import streamlit as st
import os
from rag_pipeline import RAGPipeline
import streamlit as st

@st.cache_resource
def load_index():
    # your document loading
    # chunking
    # embeddings
    # FAISS index
    return index

index = load_index()
# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .chunk-box {
        background-color: #f0f4ff;
        border-left: 4px solid #4f8ef7;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .answer-box {
        background-color: #eaffea;
        border-left: 4px solid #28a745;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 1em;
    }
    .source-tag {
        font-size: 0.75em;
        color: #888;
        margin-top: 4px;
    }
    .score-tag {
        font-size: 0.75em;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Get a free key at https://openrouter.ai",
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5)
    chunk_size = st.slider("Chunk size (chars)", min_value=100, max_value=600, value=300, step=50)

    st.divider()
    st.markdown("**Documents loaded from** `data/` folder")
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".txt"):
                st.markdown(f"📄 `{f}`")

    st.divider()
    if st.button("🔄 Rebuild Index"):
        st.session_state.pop("rag", None)
        st.rerun()

    st.markdown("""
    ---
    **How it works:**
    1. Your question is embedded
    2. Top-K similar chunks retrieved via FAISS
    3. LLM answers using ONLY those chunks
    4. No hallucination — grounded answers only
    """)


# ── Initialize RAG Pipeline ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(chunk_size: int, api_key: str) -> RAGPipeline:
    pipeline = RAGPipeline(data_dir="data", chunk_size=chunk_size, top_k=5, api_key=api_key)
    pipeline.build_index()
    return pipeline


# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("🔍 Mini RAG Chatbot")
st.caption("Ask questions — answers are grounded strictly in the loaded documents.")

# Build index
with st.spinner("Loading documents and building index..."):
    try:
        if "rag" not in st.session_state:
            st.session_state.rag = load_pipeline(chunk_size, api_key)
        rag = st.session_state.rag
        rag.top_k = top_k   # Apply live slider
    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")
        st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved"):
            with st.expander("📚 Retrieved Context Chunks", expanded=False):
                for r in msg["retrieved"]:
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<strong>Chunk #{r["rank"]}</strong> '
                        f'<span class="source-tag">Source: {r["source"]}</span> '
                        f'<span class="score-tag">| L2 score: {r["score"]:.4f}</span>'
                        f'<br><br>{r["chunk"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check API key before generating
    if not api_key:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to get AI-generated answers. Get a free key at https://openrouter.ai")
        st.stop()

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating answer..."):
            try:
                result = rag.query(prompt)

                # Show retrieved chunks
                with st.expander("📚 Retrieved Context Chunks", expanded=True):
                    for r in result["retrieved"]:
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<strong>Chunk #{r["rank"]}</strong> '
                            f'<span class="source-tag">Source: {r["source"]}</span> '
                            f'<span class="score-tag">| L2 score: {r["score"]:.4f}</span>'
                            f'<br><br>{r["chunk"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Show answer
                st.markdown("**✅ Final Answer:**")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**✅ Final Answer:**\n\n{result['answer']}",
                    "retrieved": result["retrieved"],
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")
