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
# When set, this reader is scoped to a Karakeep list: recall widens to the list's
# ancestor chain ({self + parents}) so a child SME list inherits its parents'
# knowledge. Empty = a plain collection reader (e.g. lab_memory), no list filter.
LIST_ID = os.environ.get("LIST_ID", "").strip()
KARAKEEP_API_ADDR = os.environ.get("KARAKEEP_API_ADDR", "http://karakeep.lab-memory.svc.cluster.local:3000")
KARAKEEP_API_KEY = os.environ.get("KARAKEEP_API_KEY", "")
PUBLIC_ADDR = os.environ.get("PUBLIC_ADDR", "https://karakeep.ash4d.com").rstrip("/")
QUERY_PREFIX = "query: "

_client = None
_parent_cache = None  # {list_id: parentId}, fetched once per process


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


def _parent_map() -> dict:
    """{list_id: parentId} for every Karakeep list, fetched once and cached.

    Karakeep list membership does not propagate to parents, so parent docs live
    only under their own list_id; inheritance is done here at query time.
    """
    global _parent_cache
    if _parent_cache is None:
        headers = {"Authorization": f"Bearer {KARAKEEP_API_KEY}"}
        parents, cursor = {}, None
        while True:
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(f"{KARAKEEP_API_ADDR}/api/v1/lists", params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            for L in data.get("lists", []):
                parents[L["id"]] = L.get("parentId")
            cursor = data.get("nextCursor")
            if not cursor:
                break
        _parent_cache = parents
    return _parent_cache


def ancestor_chain(list_id: str) -> list:
    """[list_id, parent, grandparent, ...] -- the scope a child inherits from."""
    chain, seen, cur = [], set(), list_id
    parents = _parent_map()
    while cur and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = parents.get(cur)
    return chain


def _scope_filter() -> str:
    """Milvus filter restricting recall to this reader's list + its ancestors."""
    if not LIST_ID:
        return ""
    ids = ancestor_chain(LIST_ID)
    quoted = ", ".join(f'"{i}"' for i in ids)
    return f"list_id in [{quoted}]"


def recall(query: str, k: int = 5, tag: str = "", expand: bool = False, per_doc: int = 2) -> list:
    """Semantic search over the shelf. `tag` filters on Karakeep tags (substring match).

    A list-scoped reader (LIST_ID set) also restricts results to its list plus all
    ancestor lists, so a child inherits parent knowledge without any data copy.
    """
    text = expand_query(query) if expand else query
    vector = embed_query(text)
    clauses = [c for c in (_scope_filter(), f'tags like "%{tag}%"' if tag else "") if c]
    flt = " and ".join(clauses)
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
                # built at query time, never read from the stored field: changing
                # PUBLIC_ADDR (e.g. http -> https) must not require re-embedding the corpus
                citation=f"{PUBLIC_ADDR}/dashboard/preview/{kid}",
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

    # A list-scoped reader files the note into its own list so the ingest worker
    # routes it to this SME collection instead of lab_memory. List membership is
    # deterministic (no async tagger in the loop), so routing is decided at save.
    if LIST_ID:
        lr = requests.put(
            f"{KARAKEEP_API_ADDR}/api/v1/lists/{LIST_ID}/bookmarks/{bid}",
            headers=headers,
            timeout=60,
        )
        lr.raise_for_status()

    return {
        "id": bid,
        "tags": wanted,
        "list_id": LIST_ID,
        "citation": f"{PUBLIC_ADDR}/dashboard/preview/{bid}",
    }
