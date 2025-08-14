## Small-model readability audit (docs and rules)

Goal: make the repo maximally usable by smaller AIs (limited context window, weaker planning). Focused on `CLAUDE.md`, `.cursor/rules/*`, and `docs/*`.

### Observed friction points
- Too much prose before the minimal, copyable command. Small models often stop early and miss the actual command.
- Variants are shown, but the shortest valid example (`{}` for tools; one-liners for scripts) is not always first or highlighted.
- Cross-references require navigation (e.g., “see Tools Documentation”). Small models benefit from inline, single-screen summaries.
- Names and roles are occasionally implicit (e.g., which node runs what). Small models prefer explicit “who/where/when”.
- Some terms require background (Hydra, extras, MCP transport). Brief parenthetical glossaries help.

### What works well already
- Clear command sections in `CLAUDE.md` and `docs/index.md` quick reference.
- Concise `.cursor/rules/*` hints; they’re short and directive.
- Tools documentation is structured and has code examples.

### Recommendations per file/category

#### 1) `CLAUDE.md`
- Add a 10-line “Small Model Quickstart” at the very top, with minimal commands only.
- Use explicit bullets for node actions: “Start server on hog”, “Tail logs for hog”, “Run align (defaults)”.
- Prefer single-line commands and avoid inline comments inside the same code block line.

Proposed preamble snippet:
```
Small Model Quickstart
- Env: `source scripts/setup_env.sh && uv pip install -e .`
- Start server (hog): `./scripts/run_mcp.sh hog`
- Tail logs (hog): `tail -f /nfs/tatbot/mcp-logs/hog.log`
- Run align (defaults): call tool `align` with `{}`
- Run stroke (tatbotlogo): call tool `stroke` with `{"scene_name":"tatbotlogo"}`
Note: Do not send ctx/context; the server provides it.
```

Rationale: places the shortest commands and the critical “don’t send ctx” guidance first.

#### 2) `.cursor/rules/*`
- Keep ultra-short; add one more line per file to make intent explicit for small models.
  - `mcp.mdc`: Add “Prefer `./scripts/run_mcp.sh <node>`; restart if tools list changes.”
  - `scripts.mdc`: Add “Prefer absolute paths; do not cd into NFS before running scripts.”
  - `uv.mdc`: Add “Use `source scripts/setup_env.sh` to avoid remembering commands.”

#### 3) `docs/index.md`
- In “Quick Reference” block, add a one-liner for “Minimal tool call via inspector” with `{}` example and a “don’t send ctx” note.
- Keep the grid cards, but small models will likely only parse the quick ref; ensure it’s comprehensive.

Proposed addition to the Quick Reference:
```
- `npx @modelcontextprotocol/inspector --config .cursor/mcp.json --server hog` then run `align` with `{}`
- Minimal tool input: always a JSON object; `{}` is valid for defaults. Never send `ctx`.
```

#### 4) `docs/mcp.md`
- Prepend a tiny “Small Model Cheatsheet” section:
```
Cheatsheet
- Start: `./scripts/run_mcp.sh hog`
- Logs: `tail -f /nfs/tatbot/mcp-logs/hog.log`
- Call tool with defaults: `align` `{}`
- If parse error: retry with `{}`; avoid strings like "[object Object]".
```
- In “Available Tools”, add a per-tool minimal call line, e.g., “align: `{}`; scene override: `{"scene_name":"default"}`”.

#### 5) `docs/tools.md`
- Add a “Minimal call pattern” box near the top:
```
Minimal Call Pattern
- Always send a JSON object as `input_data`.
- Defaults: `{}` works for all tools.
- Common keys: `scene_name` (string), `debug` (bool).
- Do not include `ctx`, `context`, `mcp_context`.
```
- Mirror the per-tool minimal example inline in each tool category.

#### 6) `docs/agent.md`
- Add a quick “validate tools capability” step and an inspector one-liner. Put warnings about “send `{}`” here too.

### Cross-cutting patterns to adopt
- Always lead with: Minimal command(s), then variants, then background.
- Use the same key names everywhere (`scene_name`); add “alias accepted: `scene`” only if implemented.
- Use bold markers for “do not send ctx/context”.
- Keep examples within 1–3 lines; prefer a single code block per task.

### Example doc rephrase (pattern)
Before:
> Tools are now defined... see Tools Documentation

After:
```
Minimal usage
- Start server (hog): `./scripts/run_mcp.sh hog`
- Call `align`: `{}`
Notes: Always send JSON object; never send ctx/context.

More: see Tools Documentation for details.
```

### Optional structural improvements
- Add `docs/quickstart_small.md` and link it from `docs/index.md` Quick Reference.
- Add `usage://<tool>` MCP resources that return a two-line quickstart for each tool.

### Validation checklist
- Confirm that a small model given only the quick reference can: start a server, tail logs, and call `align` successfully with `{}`.
- Spot-check that every tool page shows `{}` as the first example.


