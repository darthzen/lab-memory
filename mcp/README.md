# lab-memory MCP

One-call recall over the shelf, with citations.

| Tool | What it does |
|---|---|
| `recall(query, k, tag, expand)` | query → (optional `llama3.1` expansion) → `snowflake-arctic-embed2` embedding **with the `query: ` prefix** → Milvus COSINE search → passages + Karakeep citation URLs. `tag` scopes to a layer-1 project tag. |
| `save(tags, url\|text, title)` | creates a Karakeep bookmark, always tagged `claude-sourced` (docs/conventions.md). The ingest CronJob embeds it within 15 minutes. |
| `memory_status()` | collection size + which backends it is pointed at. |

## Deployment

Mirrors `ollama-code-mcp`: FastMCP over streamable-http on :8765, SUSE BCI Python,
in-cluster Deployment behind Traefik with a cert-manager DNS-01 certificate.

- `k8s/deployment.yaml` — interim: BCI python + `pip install` at start, source from the
  `lab-memory-mcp-src` ConfigMap. Replace with the Harbor-built image from `Dockerfile`.
- `k8s/service.yaml`, `k8s/ingress.yaml` — `mcp-memory.ash4d.com/mcp` (needs the DNS
  record pointing at the shared Traefik VIP, 192.168.7.150).
- `k8s/eval-job.yaml` — T-008 acceptance run (`eval_queries.py`).

Asymmetry that matters: documents are embedded raw at ingest; **only queries carry the
`query: ` prefix**. Both sides of that contract live in `retrieval.py` and `worker/ingest.py`.
