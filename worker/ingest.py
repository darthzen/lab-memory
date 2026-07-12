"""lab-memory ingest worker: stateless Karakeep -> Milvus reconcile.

Mirrors every Karakeep bookmark with usable text into the `lab_memory`
Milvus collection (chunked, embedded via Ollama/snowflake-arctic-embed2).
Safe to run repeatedly; derives all state by diffing the two systems.
Documents are embedded RAW; recall side must prepend QUERY_PREFIX.

Citation URLs are built from PUBLIC_ADDR (what a human can click), never
from the in-cluster service address used for API calls.
"""

import os
import re
import sys
import time

import requests
from pymilvus import DataType, MilvusClient

KARAKEEP_API_ADDR = os.environ.get("KARAKEEP_API_ADDR", "http://karakeep.lab-memory.svc.cluster.local:3000")
PUBLIC_ADDR = os.environ.get("PUBLIC_ADDR", "https://karakeep.ash4d.com").rstrip("/")
KARAKEEP_API_KEY = os.environ.get("KARAKEEP_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.ai.svc.cluster.local:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "snowflake-arctic-embed2")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://milvus.ai.svc.cluster.local:19530")
COLLECTION = os.environ.get("COLLECTION", "lab_memory")
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "3200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "16"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "40"))
QUERY_PREFIX = "query: "  # arctic-embed2 asymmetry: queries get this prefix, documents do not
VECTOR_DIM = 1024

PAGE_MARK = re.compile(r"-{10,}Page \(?\d+\)? ?Break-{10,}")
TAG_RE = re.compile(r"<[^>]+>")


def clip(s: str, max_bytes: int) -> str:
    """Milvus VARCHAR max_length counts BYTES. Truncate on a UTF-8 boundary."""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", "ignore")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def fetch_karakeep() -> dict:
    out = {}
    cursor = None
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {KARAKEEP_API_KEY}"
    while True:
        params = {"limit": 50, "includeContent": "true"}
        if cursor:
            params["cursor"] = cursor
        r = sess.get(f"{KARAKEEP_API_ADDR}/api/v1/bookmarks", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for bm in data.get("bookmarks", []):
            content = bm.get("content") or {}
            ctype = content.get("type")
            preview_url = f"{PUBLIC_ADDR}/dashboard/preview/{bm['id']}"
            if ctype == "link":
                text = TAG_RE.sub(" ", content.get("htmlContent") or "") or (content.get("description") or "")
                url = content.get("url") or preview_url
            elif ctype == "text":
                text = content.get("text") or ""
                url = preview_url
            elif ctype == "asset":
                text = content.get("content") or ""
                url = preview_url
            else:
                continue
            text = text.strip()
            if len(text) < MIN_TEXT_CHARS:
                print(f"SKIP {bm['id']} (text too short: {len(text)} chars)")
                continue
            out[bm["id"]] = {
                "modified_at": clip(bm.get("modifiedAt") or bm.get("createdAt") or "", 64),
                "title": clip(bm.get("title") or content.get("fileName") or bm["id"], 500),
                "tags": [t["name"] for t in bm.get("tags", [])],
                "text": text,
                "url": clip(url, 1000),
                "preview_url": clip(preview_url, 1000),
            }
        cursor = data.get("nextCursor")
        if not cursor:
            return out


def chunk_text(text: str) -> list:
    """Split on page-break markers or paragraph breaks near the window tail.

    Always makes forward progress; every returned chunk is non-empty.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        window = text[start:end]
        if end < n:
            tail_from = int(len(window) * 0.8)
            cut = -1
            last = None
            for last in PAGE_MARK.finditer(window, tail_from):
                pass
            if last is not None:
                cut = last.start()
            else:
                cut = window.rfind("\n\n", tail_from)
            if cut > 0:
                end = start + cut
                window = text[start:end]
        window = window.strip()
        if window:
            chunks.append(window)
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def embed(texts: list) -> list:
    """Embed in batches. Guarantees one vector per input, in order."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": EMBED_MODEL, "input": batch},
                    timeout=180,
                )
                r.raise_for_status()
                payload = r.json()
                got = payload.get("embeddings")
                if not isinstance(got, list) or len(got) != len(batch):
                    raise ValueError(
                        f"malformed embed response: expected {len(batch)} vectors, "
                        f"got {len(got) if isinstance(got, list) else type(got).__name__}"
                    )
                if any(len(v) != VECTOR_DIM for v in got):
                    raise ValueError(f"unexpected vector dim (want {VECTOR_DIM}) from model {EMBED_MODEL}")
                vectors.extend(got)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"WARN: embed batch retry {attempt + 1}: {exc}")
                time.sleep(5 * (attempt + 1))
    if len(vectors) != len(texts):
        raise RuntimeError(f"vector/chunk misalignment: {len(vectors)} vectors for {len(texts)} chunks")
    return vectors


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION):
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("karakeep_id", DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_ix", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("url", DataType.VARCHAR, max_length=1024)
    schema.add_field("preview_url", DataType.VARCHAR, max_length=1024)
    schema.add_field("tags", DataType.VARCHAR, max_length=1024)
    schema.add_field("modified_at", DataType.VARCHAR, max_length=64)
    schema.add_field("preview", DataType.VARCHAR, max_length=2048)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    print(f"CREATED collection {COLLECTION} (dim={VECTOR_DIM}, COSINE)")


def milvus_state(client: MilvusClient) -> dict:
    """Every karakeep_id already in Milvus -> its embedded modified_at.

    Paginated: Milvus caps a single query at 16384 rows and the corpus will
    outgrow that; iterating keeps the reconcile honest at any size.
    """
    state = {}
    it = client.query_iterator(
        collection_name=COLLECTION,
        filter="pk >= 0",
        output_fields=["karakeep_id", "modified_at"],
        batch_size=1000,
    )
    try:
        while True:
            rows = it.next()
            if not rows:
                break
            for row in rows:
                state[row["karakeep_id"]] = row["modified_at"]
    finally:
        it.close()
    return state


def main() -> None:
    if not KARAKEEP_API_KEY:
        die("KARAKEEP_API_KEY not set")
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)
    client.load_collection(COLLECTION)

    karakeep = fetch_karakeep()
    existing = milvus_state(client)
    to_upsert = [k for k, v in karakeep.items() if existing.get(k) != v["modified_at"]]
    to_delete = [k for k in existing if k not in karakeep]
    print(f"RECONCILE karakeep={len(karakeep)} milvus_docs={len(existing)} upsert={len(to_upsert)} delete={len(to_delete)}")

    failures = 0
    for kid in to_upsert:
        item = karakeep[kid]
        try:
            chunks = chunk_text(item["text"])
            if not chunks:
                print(f"SKIP {kid} (no chunks after splitting)")
                continue
            vectors = embed(chunks)
            # Delete-then-insert: a crash between the two leaves the doc absent
            # rather than duplicated, and the next run re-upserts it.
            client.delete(collection_name=COLLECTION, filter=f'karakeep_id == "{kid}"')
            rows = [
                {
                    "karakeep_id": kid,
                    "chunk_ix": ix,
                    "vector": vec,
                    "title": item["title"],
                    "url": item["url"],
                    "preview_url": item["preview_url"],
                    "tags": clip(",".join(item["tags"]), 1000),
                    "modified_at": item["modified_at"],
                    "preview": clip(chunk, 2000),
                }
                for ix, (chunk, vec) in enumerate(zip(chunks, vectors))
            ]
            client.insert(collection_name=COLLECTION, data=rows)
            print(f"UPSERT {kid} \"{item['title'][:60]}\" chunks={len(rows)}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR upserting {kid}: {exc}", file=sys.stderr)

    for kid in to_delete:
        client.delete(collection_name=COLLECTION, filter=f'karakeep_id == "{kid}"')
        print(f"DELETE {kid}")

    client.flush(COLLECTION)
    print(f"SUMMARY upserted={len(to_upsert) - failures} deleted={len(to_delete)} failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
