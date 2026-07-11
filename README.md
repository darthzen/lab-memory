# lab-memory

Joint human+AI memory for the [ash4d lab](https://github.com/darthzen/lab-fleet): a three-tier
system where each store does only what it's best at, with **all inference on local Ollama**.

| Tier | Store | Role |
|---|---|---|
| Shelf (system of record) | [Karakeep](https://karakeep.app/) | Every artifact - links, notes, PDFs, images. Provenance lives here. Human-browsable + agent-writable (MCP). |
| Semantic index | Milvus | Vectors over Karakeep content; every vector carries `karakeep_id`. Derived data only - droppable and rebuildable. |
| Working memory | Claude memory files | Small distilled facts loaded every session, citing Karakeep entries. |

Flows: **capture** -> **enrich** (CronJob: chunk -> embed -> Milvus) -> **recall** (semantic query -> `karakeep_id` -> readable source) -> **promote** (durable conclusions -> memory files).

## Model map (all local, V100)

| Job | Model |
|---|---|
| Bookmark tagging + summarization | `llama3.1` |
| Image tagging | `gemma4:12b` (multimodal) |
| Embeddings (1024-dim) | `snowflake-arctic-embed2` |
| Batch enrichment / rerank | `qwen3.6:35b` (off-hours) |
| Build-time code drafting | `qwen3-coder:30b` via [ollama-code-mcp](https://github.com/darthzen/ollama-code-mcp) |

## Status

Work is tracked on [milestones](https://github.com/darthzen/lab-memory/milestones):
P1 Karakeep deploy -> P2 ingest worker -> P3 lab-memory MCP -> P4 promotion & hygiene.

## Components (this repo)

- `deploy/` - Karakeep Helm values, ingress, namespace (P1)
- `worker/` - ingest CronJob, Python on SUSE BCI (P2)
- `mcp/` - lab-memory FastMCP server: `recall()` / `save()` (P3)
- `docs/conventions.md` - tag taxonomy + promotion criteria (P1)
