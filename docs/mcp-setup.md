# Karakeep MCP — hooking Claude into the shelf (T-005)

Official server: `@karakeep/mcp` (npm). Tools: search bookmarks, create text/URL
bookmarks, manage lists, attach/detach tags.

## Claude Desktop

Merge into `~/Library/Application Support/Claude/claude_desktop_config.json`
under `mcpServers`, then restart the app:

```json
{
  "mcpServers": {
    "karakeep": {
      "command": "npx",
      "args": ["@karakeep/mcp"],
      "env": {
        "KARAKEEP_API_ADDR": "http://karakeep.ash4d.com",
        "KARAKEEP_API_KEY": "<API key from Karakeep UI -> Settings -> API Keys>"
      }
    }
  }
}
```

## Claude Code

```bash
claude mcp add karakeep \
  -e KARAKEEP_API_ADDR=http://karakeep.ash4d.com \
  -e KARAKEEP_API_KEY=<key> \
  -- npx @karakeep/mcp
```

## Agent conventions (binding)

Any agent writing through this MCP follows [`conventions.md`](conventions.md):
one Layer-1 project tag per bookmark, `claude-sourced` on every agent-created
entry, AI freeform tags ride on top. Promotion to persistent agent memory
follows the criteria in the same doc.

Note: keys are per-user secrets. Never commit them; the working copy with the
real key lives outside git (`~/Developer/lab-memory-work/`).
