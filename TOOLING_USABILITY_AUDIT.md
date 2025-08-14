## Tooling usability audit (MCP tools)

### Executive summary
- Root cause: smaller models often send a non-JSON placeholder string (e.g., "[object Object]") or otherwise malformed input to MCP tools. Current parsing treats such strings as hard errors instead of defaulting to safe defaults. This is visible in `nfs/mcp-logs/hog.log` as repeated JSON parse failures.
- Secondary causes: verbose docstrings bury the minimal call pattern; lack of ultra-concise “call schema” up front; minor naming ambiguity; and no structured usage hints on error.
- Impact: avoidable failures when calling tools with defaults, even though tools can operate with `{}`.

### Evidence (from logs)
```
[2025-08-14 09:36:18,282] ERROR: Failed to parse input for align as JSON or python-literal dict: ...
... Invalid input for align: Expecting value: line 1 column 2 (char 1)
```
This matches clients sending the literal string "[object Object]" (typical when a JS object is implicitly stringified).

### Current implementation highlights
- Wrapper: `tatbot/src/tatbot/tools/registry.py` (`tool()` decorator creates `mcp_exposed_wrapper`).
  - Accepts `input_data: Union[str, dict, Any] | None` and forwards to `_parse_input_data`.
  - `_parse_input_data` tries `json.loads`, falls back to `ast.literal_eval`, else raises `SerializationError`.
  - Known client mistakes are logged but result in non-recoverable error instead of safe defaults.
  - Filters out problematic keys (`ctx`, `context`, etc.).

- Models: `tatbot/src/tatbot/tools/robot/models.py` and base `ToolInput`/`ToolOutput` in `tatbot/src/tatbot/tools/base.py`.
  - Fields are simple; default-able. Good for default `{}` calls.
  - No field aliases or strict per-field descriptions. Unknown extras behavior relies on Pydantic defaults.

- Docstrings: tools have long-form descriptions with examples, but minimal call (`{}`) is not the first line and can be overlooked by small models.

### Root causes
1. Fragile string parsing path: non-JSON placeholder strings (e.g., "[object Object]") produce hard failures rather than default inputs.
2. Minimal call not sufficiently prominent: examples exist but are not the first, shortest instruction.
3. No structured “usage hints” in error messages: errors do not guide clients to “send `{}` for defaults”.
4. Potential unknown-key confusion: if small models add extra keys, behavior depends on Pydantic extras config; making it explicit and permissive helps.
5. Minor naming ambiguity: short names (`align`, `sense`, `stroke`) can be confused with other domains by simple models; clarity/aliases help.

### Recommendations (ranked, low-risk first)

#### A. Be forgiving in input parsing (guardrails)
1. Treat common placeholder/empty strings as `{}` (safe defaults):
   - Strings equal to any of: `""`, `"null"`, `"None"`, `"undefined"`, `"[object Object]"`, `"[object]"` -> parse as `{}` with a warning.
   - If JSON parse fails and python-literal parse fails, degrade gracefully to `{}` instead of error, with a clear warning. Optionally gate by a strictness flag.

2. Keep filtering of framework fields (`ctx`, `context`, `mcp_context`, `tool_context`). Already implemented; keep as-is.

3. Add aliasing and extra-key handling in models:
   - On `ToolInput` (base), set `model_config = ConfigDict(extra="ignore")` to ignore unknown extras.
   - In `RobotOpInput`, add `scene: str = Field("default", alias="scene_name")` and `populate_by_name=True` so both `scene` and `scene_name` work.

4. Enrich errors with actionable usage hints:
   - When parse fails or validation fails, include a short hint like: "Send `{}` to use defaults" and "Do not send `ctx`".
   - Optionally add a `usage_hints: list[str]` to `ToolOutput` (non-breaking if optional) so UI can render structured hints.

#### B. Make the minimal call obvious (docs shape)
Adopt a compact, uniform docstring preamble for every tool (placed at the top before any paragraphs):

```
Call schema: {"scene_name?": str="default", "debug?": bool=false, ...}
Minimal valid call: {}
Examples: {"scene_name": "default"}
Notes: Do not include ctx/context; the server supplies it.
```

Apply this to `align`, `reset`, `sense`, `stroke`, `list_nodes`, and any other exposed tools. Keep longer explanations after the preamble. This reduces cognitive load for small models.

#### C. Naming and aliasing (disambiguation)
- Option 1 (non-breaking): keep current names but add very short “purpose verbs” to descriptions: e.g., "align: calibration strokes; default `{}`".
- Option 2 (optional aliases): register additional alias names that are clearer to simple models, e.g.:
  - `robot_align`, `robot_reset`, `robot_sense`, `robot_stroke`, `system_list_nodes`.
  These can coexist with current names via multiple registrations pointing to the same wrapper.

#### D. Misc small improvements
- Ensure every field has a `Field(..., description="...")` so schema annotations are concise and precise.
- Ensure example blocks always include the `{}` example on the first line.
- Consider adding a very small per-tool `usage` resource: `usage://align` responding with a 1–2 line quickstart.

### Suggested code edits (sketches)

1) `_parse_input_data` – degrade gracefully and normalize common placeholders:
```python
# tatbot/src/tatbot/tools/registry.py
def _parse_input_data(input_data: Union[str, dict, Any], model_class: type, tool_name: str) -> Any:
    if isinstance(input_data, str):
        s = input_data.strip()
        placeholder_strings = {"", "null", "None", "undefined", "[object Object]", "[object]"}
        if s in placeholder_strings or s.startswith("[object"):
            log.warning(f"{tool_name}: received placeholder string '{s}', defaulting to {{}}")
            data_dict = {}
        else:
            try:
                data_dict = json.loads(s)
            except json.JSONDecodeError as e:
                try:
                    literal = ast.literal_eval(s)
                    data_dict = literal if isinstance(literal, dict) else {}
                except Exception:
                    log.warning(f"{tool_name}: could not parse string input, defaulting to {{}}: {e}")
                    data_dict = {}
    elif isinstance(input_data, dict):
        data_dict = input_data.copy()
    elif isinstance(input_data, model_class):
        return input_data
    else:
        log.warning(f"{tool_name}: unexpected input type {type(input_data)}, defaulting to {{}}")
        data_dict = {}
    # ... then filter ctx/context and instantiate model as today
```

2) Accept unknown extras and alias `scene`:
```python
# tatbot/src/tatbot/tools/base.py
from pydantic import BaseModel, ConfigDict

class ToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    debug: bool = False

# tatbot/src/tatbot/tools/robot/models.py
from pydantic import Field, ConfigDict

class RobotOpInput(ToolInput):
    model_config = ConfigDict(populate_by_name=True)
    scene_name: str = Field("default", description="Scene config name", alias="scene")
```

3) Add concise preamble to every tool docstring (example for `align`):
```python
"""
Call schema: {"scene_name?": str="default", "debug?": bool=false}
Minimal valid call: {}
Examples: {"scene_name": "default"}
Notes: Do not include ctx/context; they are provided by the server.

Generate and execute alignment strokes for robot calibration.
... (keep the current long-form text below)
"""
```

4) Enrich error messages with hints:
```python
# in registry.mcp_exposed_wrapper except blocks
error_msg = (
    f"❌ Input validation failed for {tool_name}: {e}. "
    "Tip: call with {} for defaults; send a JSON object; do not include 'ctx'."
)
```

5) Optional aliases (non-breaking):
```python
# server._register_tools: after registering primary name
if tool_name in {"align","reset","sense","stroke","list_nodes"}:
    alias = {
        "align":"robot_align",
        "reset":"robot_reset",
        "sense":"robot_sense",
        "stroke":"robot_stroke",
        "list_nodes":"system_list_nodes",
    }[tool_name]
    mcp.tool()(available_tools[tool_name])  # register again under alias
```

### Expected outcomes
- Calling tools with missing or malformed inputs will succeed using defaults instead of failing.
- Small models will prefer the minimal `{}` example placed at the top of each docstring.
- Unknown keys will be ignored; `scene` and `scene_name` both work.
- Errors will include clear, actionable guidance.
- Optional alias names reduce ambiguity for simple planners.

### Rollout plan
1. Implement A.1–A.4 (guardrails + hints) — no interface breakage, immediate UX win.
2. Update docstring preambles across all tools.
3. Add model aliases and extra=ignore behavior.
4. Consider alias registrations after verifying they don’t confuse existing clients.

### Post-change validation
- Re-run the prior failing scenario (client sending "[object Object]") and confirm `align` succeeds with defaults and logs a warning.
- Use automated tests to call the wrappers with: `{}`, `None`, "", "[object Object]", and random extra keys; all should pass.


