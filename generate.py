"""
generate.py - LLM answer generation, grounded strictly in retrieved context.
Uses OpenRouter (OpenAI-compatible API) with Mistral-7B by default.
"""

from openai import OpenAI

MODEL = "openai/gpt-3.5-turbo" # Free model


def get_client(api_key: str) -> OpenAI:
    """
    Create OpenRouter client using provided API key.
    """
    if not api_key:
        return None

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def build_prompt(context: str, query: str) -> str:
    """
    Build a RAG prompt that always provides an answer using general knowledge when needed.
    """
    return f"""You are a helpful assistant. Answer the user's question based on the context below.
Always provide a complete answer - use the context if relevant, otherwise use your general knowledge.
Do not say "I don't know" or ask for more information. Just answer the question.

---
CONTEXT:
{context}
---

QUESTION: {query}

ANSWER:"""


def generate_answer(retrieved_chunks: list[str], query: str, api_key: str) -> str:
    """
    Generate grounded answer using OpenRouter.
    """
    context = "\n\n".join(
        f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = build_prompt(context, query)

    client = get_client(api_key)

    if client is None:
        return f"Please enter an OpenRouter API key in the sidebar to get AI-generated answers.\n\nIn the meantime, here are the retrieved document sections that may help answer your question:\n\n{context}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()