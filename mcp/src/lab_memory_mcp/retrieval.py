"""Recall + save core for lab-memory. Shared by the MCP server and the eval harness.

Retrieval path: query -> (optional llama3.1 expansion) -> arctic-embed2 with the
QUERY_PREFIX -> Milvus COSINE search -> snippets with Karakeep citations.
Documents were embedded raw at ingest time; only queries carry the prefix.
"""

import os
from dataclasses import dataclass

import requests
from pymilvus import MilvusClient

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.ai.svc.cluster.local:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "snowflake-arctic-embed2")
EXPAND_MODEL = os.environ.get("EXPAND_MODEL", "llama3.1")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://milvus.ai.svc.cluster.local:19530")
COLLECTION = os.environ.get("COLLECTION", "lab_memory")
KARAKEEP_API_ADDR = os.environ.get("KARAKEEP_API_ADDR", "http://karakeep.lab-memory.svc.cluster.local:3000")
KARAKEEP_API_KEY = os.environ.get("KARAKEEP_API_KEY", "")
PUBLIC_ADDR = os.environ.get("PUBLIC_ADDR", "http://karakeep.ash4d.com").rstrip("/")
QUERY_PREFIX = "query: "

_client = None


def milvus() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
        _client.load_collection(COLLECTION)
    return _client


@dataclass
class Hit:
    title: str
    snippet: str
    citation: str
    source_url: str
    tags: str
    score: float
    chunk_ix: int
    karakeep_id: str


def expand_query(query: str, timeout: int = 30) -> str:
    """Widen the query with synonyms via llama3.1. Falls back to the raw query."""
    prompt = (
        "Rewrite this search query for a semantic search engine over technical "
        "documents. Keep every proper noun. Add likely synonyms and the words a "
        "manual would use. Answer with the rewritten query only, one line.\n\n"
        f"Query: {query}"
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": EXPAND_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=timeout,
        )
        r.raise_for_status()
        out = (r.json().get("response") or "").strip().splitlines()
        text = out[0].strip() if out else ""
        return f"{query} {text}".strip() if text else query
    except Exception:  # noqa: BLE001 -- expansion is best-effort
        return query


def embed_query(text: str) -> list:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [QUERY_PREFIX + text]},
        timeout=120,
    )
    r.raise_for_status()
    vectors = r.json().get("embeddings")
    if not vectors:
        raise RuntimeError(f"no embedding returned by {EMBED_MODEL}")
    return vectors[0]


def recall(query: str, k: int = 5, tag: str = "", expand: bool = False, per_doc: int = 2) -> list:
    """Semantic search over the shelf. `tag` filters on Karakeep tags (substring match)."""
    text = expand_query(query) if expand else query
    vector = embed_query(text)
    flt = f'tags like "%{tag}%"' if tag else ""
    res = milvus().search(
        collection_name=COLLECTION,
        data=[vector],
        limit=max(k * 4, 12),
        filter=flt,
        output_fields=["karakeep_id", "chunk_ix", "title", "url", "preview_url", "tags", "preview"],
        search_params={"metric_type": "COSINE"},
    )
    hits = []
    seen = {}
    for match in res[0]:
        e = match["entity"]
        kid = e["karakeep_id"]
        if seen.get(kid, 0) >= per_doc:
            continue
        seen[kid] = seen.get(kid, 0) + 1
        hits.append(
            Hit(
                title=e["title"],
                snippet=e["preview"],
                citation=e.get("preview_url") or f"{PUBLIC_ADDR}/dashboard/preview/{kid}",
                source_url=e["url"],
                tags=e["tags"],
                score=round(float(match["distance"]), 4),
                chunk_ix=int(e["chunk_ix"]),
                karakeep_id=kid,
            )
        )
        if len(hits) >= k:
            break
    return hits


def save(tags: list, url: str = "", text: str = "", title: str = "") -> dict:
    """Create a Karakeep bookmark. Provenance tag `claude-sourced` is always added
    (docs/conventions.md); the ingest CronJob embeds it on the next cycle."""
    if not KARAKEEP_API_KEY:
        raise RuntimeError("KARAKEEP_API_KEY not set")
    if bool(url) == bool(text):
        raise ValueError("pass exactly one of url or text")
    headers = {"Authorization": f"Bearer {KARAKEEP_API_KEY}", "Content-Type": "application/json"}
    body = {"type": "link", "url": url} if url else {"type": "text", "text": text}
    if title:
        body["title"] = title
    r = requests.post(f"{KARAKEEP_API_ADDR}/api/v1/bookmarks", json=body, headers=headers, timeout=60)
    r.raise_for_status()
    bookmark = r.json()
    bid = bookmark["id"]

    wanted = list(dict.fromkeys([t.strip() for t in tags if t.strip()] + ["claude-sourced"]))
    tr = requests.post(
        f"{KARAKEEP_API_ADDR}/api/v1/bookmarks/{bid}/tags",
        json={"tags": [{"tagName": t} for t in wanted]},
        headers=headers,
        timeout=60,
    )
    tr.raise_for_status()
    return {
        "id": bid,
        "tags": wanted,
        "citation": f"{PUBLIC_ADDR}/dashboard/preview/{bid}",
    }
