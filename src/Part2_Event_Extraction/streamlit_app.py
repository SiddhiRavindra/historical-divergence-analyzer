import os
import json
import re
import requests
import streamlit as st
# from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

from rag_pipeline import LincolnVectorStore  # make sure this file contains the class


# -------------------------
# Env + App config
# -------------------------
# load_dotenv(override=True)

# CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
# CHROMA_TENANT = os.getenv("CHROMA_TENANT")
# CHROMA_DB = os.getenv("CHROMA_DB")
# CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "historical_divergence_part2_v1")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

CHROMA_API_KEY = get_secret("CHROMA_API_KEY")
CHROMA_TENANT = get_secret("CHROMA_TENANT")
CHROMA_DB = get_secret("CHROMA_DB")
CHROMA_COLLECTION = get_secret("CHROMA_COLLECTION", "lincoln_divergence_claims_v1")

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


st.set_page_config(page_title="Lincoln Divergence RAG", layout="wide")
st.title("Historical Divergence RAG: Lincoln vs Other Authors")


# -------------------------
# Wikipedia fallback helpers (FIXED 403)
# -------------------------
WIKI_HEADERS = {
    "User-Agent": "LincolnDivergenceRAG/1.0 (contact: kapadnis.ri@northeastern.edu)",
    "Accept-Language": "en"
}

def wiki_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search Wikipedia titles using MediaWiki API (safe, no hard crash on 403)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": top_k,
    }
    try:
        r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    results = []
    for item in data.get("query", {}).get("search", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": re.sub("<.*?>", "", item.get("snippet", ""))  # remove HTML tags
        })
    return results


def wiki_summary(title: str) -> Optional[Dict[str, Any]]:
    """Fetch a short summary using Wikipedia REST API."""
    safe_title = requests.utils.quote(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    return {
        "title": data.get("title", title),
        "extract": data.get("extract", ""),
        "url": (data.get("content_urls", {}) or {}).get("desktop", {}).get("page", "")
    }


# -------------------------
# LLM call (structured JSON output)
# -------------------------
def build_structured_prompt(
    user_query: str,
    lincoln_hits: List[Dict[str, Any]],
    others_hits: List[Dict[str, Any]],
    wiki_ctx: Optional[Dict[str, Any]]
) -> str:
    """
    Prompt asks for strict JSON. The model MUST ground in evidence;
    if missing, it must say so and rely on Wikipedia fallback ONLY if provided.
    """

    def fmt_hits(hits: List[Dict[str, Any]]) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            m = h.get("metadata", {}) or {}
            lines.append(
                f"[{i}] text={h.get('text','')}\n"
                f"    meta={json.dumps(m, ensure_ascii=False)}\n"
            )
        return "\n".join(lines).strip()

    lin_str = fmt_hits(lincoln_hits)
    oth_str = fmt_hits(others_hits)

    wiki_str = ""
    if wiki_ctx:
        wiki_str = (
            f"WikiTitle: {wiki_ctx.get('title','')}\n"
            f"WikiURL: {wiki_ctx.get('url','')}\n"
            f"WikiSummary: {wiki_ctx.get('extract','')}\n"
        )

    return f"""
You are a historical evidence analyst.
Answer the user's question using ONLY the provided evidence.
If evidence is insufficient, say so and list what is missing.
Return ONLY valid JSON matching the schema below. No markdown.

USER_QUESTION:
{user_query}

LINCOLN_EVIDENCE (top chunks):
{lin_str if lin_str else "NONE"}

OTHER_AUTHORS_EVIDENCE (top chunks):
{oth_str if oth_str else "NONE"}

WIKIPEDIA_FALLBACK (optional; use ONLY if evidence above is insufficient):
{wiki_str if wiki_str else "NONE"}

JSON_SCHEMA (must follow exactly):
{{
  "answer": "string",
  "confidence": "low|medium|high",
  "evidence_used": {{
    "lincoln": [{{"chunk_text": "string", "source_id": "string", "source_title": "string", "author": "string", "event": "string"}}],
    "others": [{{"chunk_text": "string", "source_id": "string", "source_title": "string", "author": "string", "event": "string"}}],
    "wikipedia": {{"title": "string", "url": "string", "summary_used": "string"}} | null
  }},
  "divergence": {{
    "consistency_score_0_100": 0,
    "contradictions": [
      {{
        "type": "FACTUAL|INTERPRETIVE|OMISSION",
        "severity_0_1": 0.0,
        "description": "string"
      }}
    ],
    "omissions": {{
      "missing_in_lincoln": ["string"],
      "missing_in_others": ["string"]
    }}
  }},
  "missing_information": ["string"],
  "notes": "string"
}}

Rules:
- If you cannot support a claim with evidence, do NOT state it as fact.
- Consistency score: 0=total contradiction, 100=perfect alignment. Base it on evidence overlap.
- severity_0_1 should reflect impact (0.2 minor wording; 0.8 major factual conflict).
- If Wikipedia is used, set evidence_used.wikipedia != null and keep confidence <= medium.
""".strip()


def call_openai_json(prompt: str) -> Dict[str, Any]:
    """OpenAI chat completion returning strict JSON (robust extraction)."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    text = (resp.choices[0].message.content or "").strip()

    # Robust JSON extraction
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)

    return json.loads(text)


# -------------------------
# App: connect to Chroma via your class
# -------------------------
@st.cache_resource
def load_store():
    if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DB, OPENAI_API_KEY]):
        raise RuntimeError("Missing required env vars. Set CHROMA_* and OPENAI_API_KEY in .env")

    store = LincolnVectorStore(
        chroma_api_key=CHROMA_API_KEY,
        chroma_tenant=CHROMA_TENANT,
        chroma_db=CHROMA_DB,
        openai_api_key=OPENAI_API_KEY,
        collection_name=CHROMA_COLLECTION,
    )
    return store


store = load_store()


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Top-K per side", 3, 12, 6)
event_filter = st.sidebar.text_input("Event filter (optional, exact event id)", "")
use_wiki = st.sidebar.checkbox("Use Wikipedia fallback if evidence is weak/missing", value=True)

# "Weak evidence" heuristic: if nothing returned, or distances are too high
distance_threshold = st.sidebar.slider("Weak-evidence distance threshold", 0.0, 2.0, 1.0)

query = st.text_input("Ask a question", "What did Lincoln say about the Union and war aims?")

col1, col2 = st.columns(2)


if st.button("Search"):
    # Retrieve
    lin_hits = store.search(query, top_k=top_k, event=event_filter or None, source_type="lincoln")
    oth_hits = store.search(query, top_k=top_k, event=event_filter or None, source_type="other_author")

    # Decide if evidence is weak
    def is_weak(hits: List[Dict[str, Any]]) -> bool:
        if not hits:
            return True
        dists = [h.get("distance") for h in hits if h.get("distance") is not None]
        return (len(dists) > 0 and min(dists) > distance_threshold)

    weak = is_weak(lin_hits) and is_weak(oth_hits)

    wiki_ctx = None
    if use_wiki and weak:
        with st.spinner("Wikipedia fallback: searching..."):
            titles = wiki_search(query, top_k=1)
            if titles:
                wiki_ctx = wiki_summary(titles[0]["title"])

    # Show retrieved evidence
    with col1:
        st.subheader("Lincoln Evidence")
        if not lin_hits:
            st.info("No Lincoln chunks found.")
        for h in lin_hits:
            m = h.get("metadata", {}) or {}
            with st.expander(f"{m.get('source_title','(no title)')} | event={m.get('event','')}"):
                st.caption(
                    f"author={m.get('author','')} | source_id={m.get('source_id','')} | distance={h.get('distance')}"
                )
                st.write(h.get("text", ""))

    with col2:
        st.subheader("Other Authors Evidence")
        if not oth_hits:
            st.info("No Other-author chunks found.")
        for h in oth_hits:
            m = h.get("metadata", {}) or {}
            with st.expander(f"{m.get('source_title','(no title)')} | event={m.get('event','')}"):
                st.caption(
                    f"author={m.get('author','')} | source_id={m.get('source_id','')} | distance={h.get('distance')}"
                )
                st.write(h.get("text", ""))

    if wiki_ctx:
        st.subheader("Wikipedia Fallback Context")
        st.write(f"**{wiki_ctx.get('title','')}**")
        if wiki_ctx.get("url"):
            st.write(wiki_ctx["url"])
        st.write(wiki_ctx.get("extract", ""))

    # LLM structured answer
    with st.spinner("Generating structured answer..."):
        prompt = build_structured_prompt(query, lin_hits, oth_hits, wiki_ctx)
        try:
            out = call_openai_json(prompt)
        except Exception as e:
            st.error(f"LLM JSON parse / call failed: {e}")
            st.stop()

    st.subheader("Structured Output (JSON)")
    st.json(out)

    st.subheader("Answer")
    st.write(out.get("answer", ""))

    st.subheader("Divergence Summary")
    div = out.get("divergence", {}) or {}
    st.write(f"Consistency score: **{div.get('consistency_score_0_100','?')} / 100**")

    contradictions = div.get("contradictions", []) or []
    if contradictions:
        for c in contradictions:
            st.write(f"- **{c.get('type')}** (severity {c.get('severity_0_1')}): {c.get('description')}")
    else:
        st.write("No contradictions reported.")

    missing = out.get("missing_information", []) or []
    if missing:
        st.subheader("Missing Information")
        for item in missing:
            st.write(f"- {item}")
