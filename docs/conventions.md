# lab-memory conventions — tags & promotion

Shared vocabulary for the joint human+AI memory system. Both Rick and Claude
save into Karakeep; these rules keep the shelf searchable and the trust
boundary visible. Drafted by qwen3-coder:30b (local Ollama), reviewed by Claude.

## Tag taxonomy

**Layer 1 (REQUIRED):** exactly one project tag per bookmark, kebab-case,
matching the GitHub board names:

- `nemoclaw`
- `lab-fleet`
- `ash4d-site`
- `edge`
- `lab-memory`
- `gear` (music equipment: pedal manuals, amp docs, gear reference)
- `misc` (unscoped saves)

**Layer 2 (OPTIONAL):** freeform tags for recall. AI auto-tags from Ollama ride
on top and are never curated away. Humans add whatever helps recall.

## Provenance tags

Every bookmark created by an AI agent MUST carry `claude-sourced`. Human saves
carry no provenance tag. Rationale: filterable trust boundary, auditability of
agent writes.

## Promotion criteria

A bookmark's content earns distillation into Claude's persistent memory files
when ALL of the following hold:

1. It changed a decision or unblocked work.
2. It will matter beyond the current milestone.
3. It can be stated as one or two factual sentences.

The memory entry must cite the Karakeep bookmark URL. The weekly hygiene
review surfaces candidates; promotion is deliberate, never automatic.

## Anti-patterns

- Tag sprawl in layer 1 — never invent new project tags without adding them here first.
- Saving secrets or credentials into bookmarks.
- Promoting speculation instead of confirmed facts.
- Deleting AI tags in bulk.
