"""lab-memory MCP: one-call recall over the Karakeep shelf, with citations.

Mirrors the ollama-code-mcp deployment pattern: FastMCP over streamable-http,
SUSE BCI Python image, in-cluster Deployment behind Traefik.
"""

import os

from mcp.server.fastmcp import FastMCP

from . import retrieval

mcp = FastMCP(
    "lab-memory",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8765")),
)


@mcp.tool()
def recall(query: str, k: int = 5, tag: str = "", expand: bool = False) -> dict:
    """Search the lab's joint memory (Karakeep shelf, semantically indexed in Milvus).

    Returns passages with a Karakeep citation URL for each. Use `tag` to scope to a
    project (`lab-memory`, `nemoclaw`, `lab-fleet`, `ash4d-site`, `edge`, `gear`,
    `misc`), and `expand` to widen a terse query with llama3.1 before embedding.

    If this reader is list-scoped (an SME domain), results are already restricted to
    its Karakeep list plus every ancestor list, so a child inherits parent knowledge.
    """
    hits = retrieval.recall(query=query, k=k, tag=tag, expand=expand)
    return {
        "query": query,
        "count": len(hits),
        "results": [
            {
                "title": h.title,
                "snippet": h.snippet,
                "citation": h.citation,
                "source_url": h.source_url,
                "tags": h.tags,
                "score": h.score,
                "chunk": h.chunk_ix,
            }
            for h in hits
        ],
    }


@mcp.tool()
def save(tags: list[str], url: str = "", text: str = "", title: str = "") -> dict:
    """Save a link or a note to the shelf (Karakeep), so it becomes recallable.

    Pass exactly one of `url` or `text`. `tags` must include one layer-1 project tag
    per docs/conventions.md; `claude-sourced` is added automatically. The ingest
    CronJob embeds the bookmark into Milvus on its next cycle (<= 15 min).
    """
    return retrieval.save(tags=tags, url=url, text=text, title=title)


@mcp.tool()
def memory_status() -> dict:
    """Health of the memory stack: collection size and reachable backends."""
    client = retrieval.milvus()
    stats = client.get_collection_stats(retrieval.COLLECTION)
    status = {
        "collection": retrieval.COLLECTION,
        "chunks": stats.get("row_count"),
        "embed_model": retrieval.EMBED_MODEL,
        "milvus": retrieval.MILVUS_URI,
        "karakeep": retrieval.PUBLIC_ADDR,
    }
    if retrieval.LIST_ID:
        status["list_scope"] = retrieval.ancestor_chain(retrieval.LIST_ID)
    return status


def main() -> None:
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "streamable-http"))


if __name__ == "__main__":
    main()
