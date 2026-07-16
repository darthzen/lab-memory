"""lab-memory ingest worker: stateless Karakeep -> Milvus reconcile.

Default: mirrors every Karakeep bookmark with usable text into the `lab_memory`
Milvus collection (chunked, embedded via Ollama/snowflake-arctic-embed2).

SME routing (SME_ROUTES): bookmarks that live under a configured Karakeep list
subtree are routed OUT of `lab_memory` and INTO a per-domain SME collection,
stamped with the `list_id` of the deepest routed list they belong to. Recall on
the SME side widens a query to {self + ancestors}, so a child list inherits its
parents' knowledge without the data ever being copied. One fetch snapshot drives
every collection in a single pass, so `lab_memory` and the SME collections never
transiently hold the same document (no comingling, no inter-job race).

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
DEFAULT_COLLECTION = os.environ.get("COLLECTION", "lab_memory")
# SME routing map: "rootListId:collection,rootListId2:collection2". Every list at
# or below a root routes its bookmarks into that root's collection. Empty = the
# worker behaves exactly as before (everything -> DEFAULT_COLLECTION).
SME_ROUTES = os.environ.get("SME_ROUTES", "")
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


def session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {KARAKEEP_API_KEY}"
    return s


def fetch_karakeep(sess: requests.Session) -> dict:
    out = {}
    cursor = None
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


# --------------------------------------------------------------------------- #
# SME list routing
# --------------------------------------------------------------------------- #

def parse_routes(spec: str) -> dict:
    """"rootListId:collection,..." -> {rootListId: collection}."""
    routes = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rid, _, coll = part.partition(":")
        rid, coll = rid.strip(), coll.strip()
        if not rid or not coll:
            die(f"malformed SME_ROUTES entry {part!r} (want rootListId:collection)")
        routes[rid] = coll
    return routes


def fetch_lists(sess: requests.Session) -> dict:
    """All Karakeep lists -> {id: {'name':..., 'parentId':...}}. One or few calls."""
    lists = {}
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = sess.get(f"{KARAKEEP_API_ADDR}/api/v1/lists", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for L in data.get("lists", []):
            lists[L["id"]] = {"name": L.get("name"), "parentId": L.get("parentId")}
        cursor = data.get("nextCursor")
        if not cursor:
            return lists


def fetch_list_members(sess: requests.Session, list_id: str) -> set:
    """Explicit bookmark ids in a list. Membership does NOT propagate to parents,
    so each list is enumerated on its own (verified against Karakeep v1)."""
    ids = set()
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = sess.get(f"{KARAKEEP_API_ADDR}/api/v1/lists/{list_id}/bookmarks", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for bm in data.get("bookmarks", []):
            ids.add(bm["id"])
        cursor = data.get("nextCursor")
        if not cursor:
            return ids


def _children(lists: dict) -> dict:
    kids = {}
    for lid, meta in lists.items():
        p = meta.get("parentId")
        if p:
            kids.setdefault(p, []).append(lid)
    return kids


def _subtree(root: str, kids: dict) -> list:
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(kids.get(n, []))
    return out


def _depth(lid: str, lists: dict) -> int:
    d, cur = 0, lists.get(lid, {}).get("parentId")
    while cur:
        d += 1
        cur = lists.get(cur, {}).get("parentId")
    return d


def resolve_routing(sess: requests.Session, routes: dict, lists: dict) -> dict:
    """bookmark_id -> (collection, list_id) for everything under a routed subtree.

    A bookmark filed in several routed lists is stamped with the DEEPEST one, so a
    doc placed in both a parent and a child is owned by the child. Raises on any
    error: a partial resolution could let an SME doc fall through to lab_memory,
    so routing is all-or-nothing to guarantee no comingling.
    """
    kids = _children(lists)
    # bookmark_id -> (collection, list_id, depth)
    best = {}
    for root, coll in routes.items():
        if root not in lists:
            die(f"SME_ROUTES root {root} not found in Karakeep lists")
        for lid in _subtree(root, kids):
            depth = _depth(lid, lists)
            for bid in fetch_list_members(sess, lid):
                prev = best.get(bid)
                if prev is None or depth > prev[2]:
                    best[bid] = (coll, lid, depth)
    return {bid: (c, l) for bid, (c, l, _d) in best.items()}


# --------------------------------------------------------------------------- #
# Chunk / embed
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Milvus
# --------------------------------------------------------------------------- #

def ensure_collection(client: MilvusClient, collection: str, with_list_id: bool) -> None:
    if client.has_collection(collection):
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
    if with_list_id:
        # owning Karakeep list; recall filters `list_id IN {self + ancestors}`
        schema.add_field("list_id", DataType.VARCHAR, max_length=32)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection, schema=schema, index_params=index_params)
    print(f"CREATED collection {collection} (dim={VECTOR_DIM}, COSINE, list_id={with_list_id})")


def milvus_state(client: MilvusClient, collection: str) -> dict:
    """Every karakeep_id already in `collection` -> its embedded modified_at.

    Paginated: Milvus caps a single query at 16384 rows and the corpus will
    outgrow that; iterating keeps the reconcile honest at any size.
    """
    state = {}
    it = client.query_iterator(
        collection_name=collection,
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


def reconcile(client: MilvusClient, collection: str, subset: dict, list_ids: dict) -> int:
    """Bring `collection` in line with `subset` (its slice of the fetch snapshot).

    `subset` is the set of bookmarks that belong in THIS collection; anything in
    the collection but absent from `subset` is deleted -- which is exactly how a
    bookmark newly filed into an SME list migrates out of lab_memory on the same
    run it lands in the SME collection. `list_ids` (bookmark_id -> list_id) is
    stamped only when the collection carries the field.
    """
    with_list_id = collection != DEFAULT_COLLECTION
    ensure_collection(client, collection, with_list_id)
    client.load_collection(collection)

    existing = milvus_state(client, collection)
    to_upsert = [k for k, v in subset.items() if existing.get(k) != v["modified_at"]]
    to_delete = [k for k in existing if k not in subset]
    print(f"RECONCILE[{collection}] karakeep={len(subset)} milvus_docs={len(existing)} "
          f"upsert={len(to_upsert)} delete={len(to_delete)}")

    failures = 0
    for kid in to_upsert:
        item = subset[kid]
        try:
            chunks = chunk_text(item["text"])
            if not chunks:
                print(f"SKIP {kid} (no chunks after splitting)")
                continue
            vectors = embed(chunks)
            # Delete-then-insert: a crash between the two leaves the doc absent
            # rather than duplicated, and the next run re-upserts it.
            client.delete(collection_name=collection, filter=f'karakeep_id == "{kid}"')
            rows = []
            for ix, (chunk, vec) in enumerate(zip(chunks, vectors)):
                row = {
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
                if with_list_id:
                    row["list_id"] = list_ids.get(kid, "")
                rows.append(row)
            client.insert(collection_name=collection, data=rows)
            print(f"UPSERT[{collection}] {kid} \"{item['title'][:60]}\" chunks={len(rows)}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR upserting {kid} -> {collection}: {exc}", file=sys.stderr)

    for kid in to_delete:
        client.delete(collection_name=collection, filter=f'karakeep_id == "{kid}"')
        print(f"DELETE[{collection}] {kid}")

    client.flush(collection)
    print(f"SUMMARY[{collection}] upserted={len(to_upsert) - failures} "
          f"deleted={len(to_delete)} failures={failures}")
    return failures


def main() -> None:
    if not KARAKEEP_API_KEY:
        die("KARAKEEP_API_KEY not set")
    sess = session()
    client = MilvusClient(uri=MILVUS_URI)

    karakeep = fetch_karakeep(sess)

    # Resolve list routing over the SAME snapshot. All-or-nothing: a failure here
    # aborts before any write, so an unresolved SME doc can never leak into
    # lab_memory.
    routes = parse_routes(SME_ROUTES)
    routing = {}
    if routes:
        lists = fetch_lists(sess)
        routing = resolve_routing(sess, routes, lists)
        print(f"ROUTING roots={list(routes.items())} routed_bookmarks={len(routing)}")

    # Partition the snapshot: routed bookmarks go to their SME collection, the
    # rest to DEFAULT_COLLECTION. Because partitions are disjoint, no document is
    # ever a member of two collections' subsets -> no comingling. Every configured
    # SME collection is seeded empty so it (and its readers) exist even before the
    # first bookmark is filed into that domain.
    partitions = {DEFAULT_COLLECTION: {}}
    for coll in routes.values():
        partitions.setdefault(coll, {})
    list_ids = {}
    for kid, item in karakeep.items():
        target = routing.get(kid)
        if target:
            coll, lid = target
            partitions.setdefault(coll, {})[kid] = item
            list_ids[kid] = lid
        else:
            partitions[DEFAULT_COLLECTION][kid] = item

    failures = 0
    for collection, subset in partitions.items():
        failures += reconcile(client, collection, subset, list_ids)

    print(f"DONE collections={len(partitions)} total_failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
