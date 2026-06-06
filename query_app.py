"""
Event Intelligence RAG System — Interactive Query App
Trinity Mobility Pvt Ltd

Run this file to ask questions about incidents in plain English.
Usage: python query_app.py
"""

import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ── CONFIG — update these paths ──────────────────────────
INDEX_PATH   = r"C:\Users\mural\Desktop\GreatLearning\Mini Projects\my_project\vector_index.pkl"
API_KEY_PLACEHOLDER = "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"   
# ─────────────────────────────────────────────────────────


def load_index(path):
    print("Loading vector index...")
    with open(path, "rb") as f:
        return pickle.load(f)


def hyde_expand(query):
    return (
        "Operational incident report: "
        "The following alarm and incident data was recorded in the emergency operations platform. "
        + query
    )


def extract_ids(query):
    """Extract any EVENT IDs (INCxxxxxx) or ALARM IDs from the query."""
    event_ids = re.findall(r'INC\d+', query.upper())
    alarm_ids = re.findall(r'\b\d{4,6}\b', query)
    return event_ids, alarm_ids


def retrieve(query, index, top_k=5):
    vectorizer = index["vectorizer"]
    matrix     = index["matrix"]
    chunks     = index["chunks"]

    # ── SMART RETRIEVAL ──────────────────────────────────
    # If query contains a specific Event ID like INC001572,
    # do EXACT match first — much more accurate than TF-IDF
    event_ids, alarm_ids = extract_ids(query)

    exact_results = []
    if event_ids:
        for chunk in chunks:
            for eid in event_ids:
                if chunk.get("event_id", "").upper() == eid:
                    exact_results.append({
                        "text":  chunk["text"],
                        "meta":  chunk,
                        "score": 1.0,   # perfect match
                    })
        if exact_results:
            print(f"  → Exact match found for {event_ids}")
            return exact_results[:top_k]

    # ── SEMANTIC RETRIEVAL (TF-IDF + cosine similarity) ──
    expanded = hyde_expand(query)
    q_vec    = vectorizer.transform([expanded])
    scores   = cosine_similarity(q_vec, matrix).flatten()
    top_idx  = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] > 0:
            results.append({
                "text":  chunks[idx]["text"],
                "meta":  chunks[idx],
                "score": round(float(scores[idx]), 4),
            })
    return results


def ask_llm(query, chunks):
    context = ""
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        context += (
            f"\n--- Context {i} ---\n"
            f"Event: {m.get('event_id','?')} | Alarm: {m.get('alarm_id','?')} | "
            f"Priority: {m.get('priority','?')} | Component: {m.get('component_id','?')}\n"
            f"{c['text']}\n"
        )

    prompt = f"""You are an Event Intelligence Assistant for an emergency operations platform.
Answer the question STRICTLY from the context below. Do NOT make up information.
If context is not enough, say: "Insufficient data in retrieved context."
Be factual, concise, and reference Event IDs when available.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY_PLACEHOLDER}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": "You are a factual emergency operations assistant. Answer only from context provided."},
                {"role": "user",   "content": prompt}
            ]
        }
    )

    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return f"[API Error {resp.status_code}]: {resp.text[:200]}"


def main():
    print("=" * 60)
    print("  EVENT INTELLIGENCE RAG SYSTEM")
    print("  Trinity Mobility Pvt Ltd")
    print("=" * 60)
    print("Loading system... please wait...")

    index = load_index(INDEX_PATH)

    print("✓ System ready! Ask questions about incidents.")
    print("  Type 'exit' or 'quit' to stop.\n")
    print("-" * 60)
    print("Example questions you can ask:")
    print("  - Give me details on incident INC001572")
    print("  - Why are there so many critical alarms from component 103?")
    print("  - Which component has the highest number of alarms?")
    print("  - What incidents involve Civil Defence as secondary agency?")
    print("  - How many High priority events are there?")
    print("-" * 60)

    while True:
        print()
        user_input = input("Your Question: ").strip()

        if not user_input:
            print("  Please type a question.")
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        print("\nSearching incidents...")
        chunks = retrieve(user_input, index, top_k=5)

        if not chunks:
            print("No relevant records found. Try rephrasing your question.")
            continue

        print(f"Found {len(chunks)} relevant records:")
        for c in chunks:
            m = c["meta"]
            print(f"  [{c['score']:.4f}] {m.get('event_id')} | "
                  f"Alarm {m.get('alarm_id')} | "
                  f"{m.get('priority')} | "
                  f"Comp {m.get('component_id')}")

        print("\nGenerating answer...")
        answer = ask_llm(user_input, chunks)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
