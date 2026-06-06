"""
Event Intelligence RAG System — Final Version
Trinity Mobility Pvt Ltd Assignment

Steps:
  1. Load CSV → SQLite
  2. Feature Engineering (event_text narrative)
  3. Text Chunking
  4. TF-IDF Embeddings + ChromaDB (offline-compatible)
  5. Retrieval (cosine similarity, Top-5, HyDE expansion)
  6. RAG with Anthropic Claude API
"""

import csv, sqlite3, json, os, pickle
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv.field_size_limit(10**7)

# ──────────────────────────────────────────────
# STEP 1: DATA INGESTION & SQL SETUP
# ──────────────────────────────────────────────

def step1_load_csv_to_sqlite(csv_path, db_path="event_intelligence.db"):
    print("\n" + "="*60)
    print("STEP 1: DATA INGESTION & SQL SETUP")
    print("="*60)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS event_details")

    typed = {
        "ALARM_ID": "INTEGER", "PRIORITY_ID": "INTEGER",
        "COMPONENT_ID": "INTEGER", "BPM_ESCULATION_COUNT": "INTEGER",
        "SOPTOTALCOUNT": "INTEGER", "SOPCOMPLETEDCOUNT": "INTEGER",
        "ALL_NOTIFICATION_COUNT": "INTEGER", "ALARM_STATUS": "INTEGER",
        "ALARM_TYPE_ID": "INTEGER", "SOURCE_TYPE_ID": "INTEGER",
        "LATITUDE": "REAL", "LONGITUDE": "REAL",
    }
    col_defs = ", ".join(f'"{c}" {typed.get(c, "TEXT")}' for c in columns)
    cur.execute(f"CREATE TABLE event_details ({col_defs})")

    placeholders = ", ".join(["?"] * len(columns))
    for row in rows:
        vals = [row.get(c) or None for c in columns]
        cur.execute(f"INSERT INTO event_details VALUES ({placeholders})", vals)

    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM event_details").fetchone()[0]
    print(f"✓ {count} rows loaded into 'event_details' ({len(columns)} columns)")
    print(f"✓ Database: {db_path}")

    print("\n--- Priority Distribution ---")
    for r in cur.execute("SELECT PRIORITY, COUNT(*) FROM event_details GROUP BY PRIORITY ORDER BY COUNT(*) DESC"):
        print(f"   {r[0] or 'NULL'}: {r[1]}")

    print("\n--- Component Distribution (top 10) ---")
    for r in cur.execute("SELECT COMPONENT_ID, COUNT(*) FROM event_details GROUP BY COMPONENT_ID ORDER BY COUNT(*) DESC LIMIT 10"):
        print(f"   Component {r[0] or 'NULL'}: {r[1]}")

    return conn


# ──────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────

def safe(row, key):
    v = row.get(key)
    return str(v).strip() if v is not None else ""

def create_event_text(row):
    """Build a rich human-readable narrative for each incident."""
    event_id   = safe(row, "EVENT_ID")
    alarm_id   = safe(row, "ALARM_ID")
    alarm_name = safe(row, "ALARM_NAME")
    priority   = safe(row, "PRIORITY")
    component  = safe(row, "COMPONENT_ID")
    severity   = safe(row, "SEVERITY")
    urgency    = safe(row, "URGENCY")
    location   = safe(row, "LOCATION")
    site_name  = safe(row, "SITE_NAME")
    juri_name  = safe(row, "JURISDICTION_NAME")
    status     = safe(row, "EVENT_STATUS") or safe(row, "ALARM_STATUS")
    alarm_time = safe(row, "ALARM_GENERATED_TIME")
    occur_time = safe(row, "EVENT_OCCURRENCE_TIME")
    sop_name   = safe(row, "SOP_NAME")
    sop_desc   = safe(row, "SOP_DESCRIPTION")
    sop_url    = safe(row, "SOP_DOCUMENT_URL")
    agency_pri = safe(row, "PRIMARY_AGENCY")
    agency_sec = safe(row, "SECONDARY_AGENCY")
    device     = safe(row, "DEVICE_NAME")
    dev_type   = safe(row, "DEVICE_TYPE_NAME") or safe(row, "DEVICE_TYPE")
    category   = safe(row, "CATEGORY_NAME") or safe(row, "ALARM_NAME")
    reason     = safe(row, "EVENT_REASON")
    step_name  = safe(row, "STEP_NAME")
    name_ctct  = safe(row, "NAME_CONTACT")
    phone      = safe(row, "PHONE")
    extra      = safe(row, "EXTRA_DETAILS_5")
    add_det    = safe(row, "ADDITIONAL_DETAILS")
    rec_data   = safe(row, "REC_DATA")
    entity     = safe(row, "ENTITY_NAME")
    lat        = safe(row, "LATITUDE")
    lon        = safe(row, "LONGITUDE")
    sop_close  = safe(row, "SOP_CLOSE_TIME")
    user_name  = safe(row, "USER_NAME")
    complaint  = safe(row, "COMPLAINT_ID")
    bpm_count  = safe(row, "BPM_ESCULATION_COUNT")
    sop_total  = safe(row, "SOPTOTALCOUNT")
    sop_done   = safe(row, "SOPCOMPLETEDCOUNT")

    parts = []
    p = f"Incident {event_id} (Alarm ID: {alarm_id}, Complaint: {complaint}) is a {priority}-priority event"
    if category:      p += f" categorized as '{category}'"
    if alarm_name and alarm_name != category: p += f" triggered by '{alarm_name}'"
    parts.append(p + ".")

    loc_parts = [x for x in [site_name, location, juri_name] if x and x not in ("NA", "null", "None")]
    if loc_parts: parts.append(f"Location: {', '.join(loc_parts)}.")
    if lat and lon: parts.append(f"Coordinates: Lat {lat}, Lon {lon}.")
    if alarm_time:   parts.append(f"Alarm generated at: {alarm_time}.")
    if occur_time:   parts.append(f"Event occurrence time: {occur_time}.")

    status_map = {"1":"Closed","2":"Resolved","3":"In Progress","5":"Pending","In progress":"In Progress"}
    parts.append(f"Status: {status_map.get(status, status) or 'Unknown'}.")
    if severity:  parts.append(f"Severity: {severity}.")
    if urgency:   parts.append(f"Urgency: {urgency}.")
    if component: parts.append(f"Component ID: {component}.")
    if device:    parts.append(f"Device: {device} (Type: {dev_type}).")
    if agency_pri: parts.append(f"Primary agency: {agency_pri}.")
    if agency_sec: parts.append(f"Secondary agencies: {agency_sec}.")
    if sop_name:  parts.append(f"SOP: '{sop_name}'.")
    if sop_desc:  parts.append(f"SOP Description: {sop_desc[:300]}.")
    if sop_url:   parts.append(f"SOP URL: {sop_url}.")
    if sop_total and sop_done: parts.append(f"SOP progress: {sop_done}/{sop_total} steps.")
    if sop_close: parts.append(f"SOP closed at: {sop_close}.")
    if step_name: parts.append(f"Current SOP step: {step_name}.")
    if name_ctct: parts.append(f"Contact: {name_ctct}.")
    if phone and phone not in ("NA",""):  parts.append(f"Phone: {phone}.")
    if user_name: parts.append(f"Assigned user: {user_name}.")
    if entity:    parts.append(f"Entity: {entity}.")
    if reason:    parts.append(f"Event reason: {reason}.")
    if extra:     parts.append(f"Extra details: {extra[:200]}.")
    if add_det:   parts.append(f"Additional info: {add_det[:200]}.")
    if rec_data:  parts.append(f"Recorded data: {rec_data[:200]}.")
    if bpm_count and bpm_count not in ("","0","None"): parts.append(f"BPM escalation count: {bpm_count}.")

    return " ".join(parts)


def step2_feature_engineering(conn):
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)

    cur = conn.cursor()
    cur.execute("SELECT * FROM event_details")
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    enriched = []
    for row in rows:
        event_text = create_event_text(row)
        month = ""
        try:
            dt = datetime.strptime(str(row.get("ALARM_GENERATED_TIME",""))[:19], "%Y-%m-%d %H:%M:%S")
            month = dt.strftime("%B %Y")
        except: pass

        enriched.append({
            "alarm_id":        str(row.get("ALARM_ID","")),
            "event_id":        str(row.get("EVENT_ID","")),
            "priority":        str(row.get("PRIORITY","") or ""),
            "component_id":    str(row.get("COMPONENT_ID","") or ""),
            "severity":        str(row.get("SEVERITY","") or ""),
            "urgency":         str(row.get("URGENCY","") or ""),
            "month":           month,
            "location":        str(row.get("LOCATION","") or ""),
            "site_name":       str(row.get("SITE_NAME","") or ""),
            "alarm_status":    str(row.get("ALARM_STATUS","") or ""),
            "event_status":    str(row.get("EVENT_STATUS","") or ""),
            "alarm_name":      str(row.get("ALARM_NAME","") or ""),
            "primary_agency":  str(row.get("PRIMARY_AGENCY","") or ""),
            "secondary_agency":str(row.get("SECONDARY_AGENCY","") or ""),
            "sop_name":        str(row.get("SOP_NAME","") or ""),
            "event_text":      event_text,
        })

    print(f"✓ {len(enriched)} records enriched with event_text narratives")
    print(f"\nSample event_text:\n{enriched[0]['event_text'][:500]}...")
    return enriched


# ──────────────────────────────────────────────
# STEP 3: TEXT CHUNKING
# ──────────────────────────────────────────────

def step3_chunk_all(enriched, chunk_size=600, overlap=60):
    print("\n" + "="*60)
    print("STEP 3: TEXT CHUNKING")
    print("="*60)

    all_chunks = []
    for rec in enriched:
        words = rec["event_text"].split()
        if len(words) <= chunk_size:
            chunks_for_rec = [{"chunk_id": f"{rec['alarm_id']}_0", **rec, "text": rec["event_text"]}]
        else:
            chunks_for_rec, start, idx = [], 0, 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk = {**rec, "chunk_id": f"{rec['alarm_id']}_{idx}", "text": " ".join(words[start:end])}
                chunks_for_rec.append(chunk)
                start += chunk_size - overlap
                idx += 1
        all_chunks.extend(chunks_for_rec)

    print(f"✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Metadata per chunk: alarm_id, event_id, priority, component_id + 10 more fields")
    return all_chunks


# ──────────────────────────────────────────────
# STEP 4: TF-IDF EMBEDDINGS + VECTOR INDEX
# ──────────────────────────────────────────────

def step4_embed_and_store(chunks, index_path="vector_index.pkl"):
    """
    Use TF-IDF vectorizer (offline, no GPU needed) as embedding model.
    Stores: vectorizer, matrix, and chunk metadata.
    TF-IDF is interpretable and well-suited for keyword-heavy operational logs.
    """
    print("\n" + "="*60)
    print("STEP 4: TF-IDF EMBEDDINGS & VECTOR INDEX")
    print("="*60)

    texts = [c["text"] for c in chunks]
    print(f"Fitting TF-IDF vectorizer on {len(texts)} chunks...")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams for better semantic capture
        max_features=30000,      # top 30K features
        sublinear_tf=True,       # log-scaled TF for better balance
        min_df=2,                # ignore extremely rare terms
        strip_accents='unicode',
    )
    matrix = vectorizer.fit_transform(texts)
    
    index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"✓ TF-IDF matrix shape: {matrix.shape}")
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"✓ Vector index saved: {index_path}")
    return index


def load_index(index_path="vector_index.pkl"):
    with open(index_path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# STEP 5: RETRIEVAL (HyDE + cosine similarity)
# ──────────────────────────────────────────────

def hyde_expand_query(query):
    """
    HyDE: Hypothetical Document Embedding.
    Expand the query into a hypothetical incident report.
    This bridges the vocabulary gap between questions and incident logs.
    """
    return (
        "Operational incident event report: "
        "The following alarm and incident data was recorded in the emergency operations platform. "
        + query
    )


def step5_retrieve(query, index, top_k=5, filter_fn=None):
    """
    Retrieve top-k chunks by cosine similarity.
    filter_fn: optional callable(chunk) -> bool for metadata filtering
    """
    vectorizer = index["vectorizer"]
    matrix     = index["matrix"]
    chunks     = index["chunks"]

    expanded = hyde_expand_query(query)
    q_vec    = vectorizer.transform([expanded])
    scores   = cosine_similarity(q_vec, matrix).flatten()

    # Apply metadata filter if provided
    if filter_fn:
        for i, chunk in enumerate(chunks):
            if not filter_fn(chunk):
                scores[i] = -1

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text":     chunks[idx]["text"],
                "metadata": {k: v for k, v in chunks[idx].items() if k != "text" and k != "event_text"},
                "score":    round(float(scores[idx]), 4),
            })
    return results


# ──────────────────────────────────────────────
# STEP 6: RAG PROMPT + LLM
# ──────────────────────────────────────────────

def build_rag_prompt(query, retrieved_chunks):
    """ReAct + Few-Shot style RAG prompt grounded in retrieved context."""
    ctx = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        m = chunk["metadata"]
        ctx += (
            f"\n--- Context {i} ---\n"
            f"Event: {m.get('event_id','?')} | Alarm: {m.get('alarm_id','?')} | "
            f"Priority: {m.get('priority','?')} | Component: {m.get('component_id','?')} | "
            f"Similarity Score: {chunk['score']}\n"
            f"{chunk['text']}\n"
        )

    return f"""You are an Event Intelligence Assistant for an emergency operations platform (Fire, EMS, Civil Defence).

Answer the question STRICTLY from the provided context. Do NOT hallucinate.
If context is insufficient, say: "Insufficient data in retrieved context."
Be concise, factual, and operational. Reference Event IDs and Alarm IDs.

--- RETRIEVED CONTEXT ---
{ctx}
--- END CONTEXT ---

User Question: {query}

Answer:"""


def query_rag_system(query, index, top_k=5, filter_fn=None, verbose=True):
    """Full RAG pipeline: retrieve → prompt → LLM → answer."""
    if verbose:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

    chunks = step5_retrieve(query, index, top_k=top_k, filter_fn=filter_fn)

    if verbose:
        print(f"Retrieved {len(chunks)} chunks:")
        for c in chunks:
            m = c["metadata"]
            print(f"  [{c['score']:.4f}] {m.get('event_id')} | Alarm {m.get('alarm_id')} | "
                  f"{m.get('priority')} | Comp {m.get('component_id')}")

    prompt = build_rag_prompt(query, chunks)

    import requests

    API_KEY_PLACEHOLDER = "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"   # <-- paste your key from console.groq.com

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY_PLACEHOLDER}",
        },
        json={
            "model": "llama-3.3-70b-versatile",   # free, 128K context window
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a factual emergency operations intelligence assistant. Answer only from the provided context. Do not hallucinate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    if resp.status_code == 200:
        answer = resp.json()["choices"][0]["message"]["content"]
    else:
        answer = f"[API Error {resp.status_code}]: {resp.text[:300]}"

    if verbose:
        print(f"\nAnswer:\n{answer}")

    return {"query": query, "retrieved_chunks": chunks, "answer": answer}


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH = r"C:\Users\mural\Desktop\GreatLearning\Mini Projects\my_project\V_EVENT_DETAILS_202512311554.csv"
    DB_PATH = r"C:\Users\mural\Desktop\GreatLearning\Mini Projects\my_project\event_intelligence.db"
    INDEX_PATH = r"C:\Users\mural\Desktop\GreatLearning\Mini Projects\my_project\vector_index.pkl"

    print("="*60)
    print(" EVENT INTELLIGENCE RAG SYSTEM")
    print(" Trinity Mobility Pvt Ltd")
    print("="*60)

    conn     = step1_load_csv_to_sqlite(CSV_PATH, DB_PATH)
    enriched = step2_feature_engineering(conn)
    chunks   = step3_chunk_all(enriched)
    index    = step4_embed_and_store(chunks, INDEX_PATH)

    print("\n" + "="*60)
    print("STEP 5 & 6: RETRIEVAL + RAG DEMO")
    print("="*60)

    test_queries = [
        "Why are there so many critical alarms from component 103?",
        "How many critical priority events are In Progress?",
        "What incidents involve Civil Defence as secondary agency?",
        "What are the SOP steps for Fire Emergency events?",
        "Which component has the highest number of alarms?",
    ]

    all_results = []
    for q in test_queries:
        result = query_rag_system(q, index, top_k=5)
        all_results.append({
            "query":   result["query"],
            "answer":  result["answer"],
            "retrieved": [
                {"event_id": c["metadata"].get("event_id"),
                 "alarm_id": c["metadata"].get("alarm_id"),
                 "score":    c["score"],
                 "preview":  c["text"][:200]}
                for c in result["retrieved_chunks"]
            ]
        })

    with open(r"C:\Users\mural\Desktop\GreatLearning\Mini Projects\my_project\rag_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print(f"  SQLite DB  : {DB_PATH}")
    print(f"  Vector Index: {INDEX_PATH}")
    print(f"  Results    : /home/claude/rag_results.json")
