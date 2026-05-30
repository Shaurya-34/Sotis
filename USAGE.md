# Sotis — Usage Guide

CLI commands, the Claude Code (MCP) integration, and the live dashboard.

> Library usage (the `SotisGuard` API) lives in the [README](README.md).
> Architecture and design decisions live in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Installation

```bash
pip install sotis

# with the dashboard
pip install sotis[obs]

# with the MCP server (Claude Code / Desktop)
pip install sotis[mcp]
```

---

## CLI Commands

```bash
sotis dashboard    # Launch the Streamlit observability dashboard
sotis benchmark    # Run the empirical benchmark suite
sotis demo         # Run the built-in meltdown/recovery demo
sotis mcp          # Run the MCP server (stdio) for Claude Code / Desktop
```

| Command | What it does |
|---------|--------------|
| `sotis dashboard` | Opens the live telemetry dashboard. Reads `logs/session_*.json` and renders entropy, meltdowns, subtask progress, and the GDS score. Toggle **Live Mode** to auto-refresh during an active session. |
| `sotis benchmark` | Runs the empirical benchmark suite and writes session logs the dashboard can replay. |
| `sotis demo` | Runs the built-in meltdown → intercept → recovery demo. |
| `sotis mcp` | Starts the MCP server over stdio so an MCP-capable agent can report tool calls and receive meltdown verdicts. |

---

## Claude Code (MCP)

Sotis ships an MCP (Model Context Protocol) server so any MCP-capable agent —
Claude Code, Claude Desktop — can report its tool calls and get live meltdown
verdicts, while streaming the same telemetry to the dashboard.

### 1. Install with the MCP extra

```bash
pip install sotis[mcp]
```

### 2. Register the server

Add it to your project's `.mcp.json` (or your Claude Desktop config):

```json
{
  "mcpServers": {
    "sotis": { "command": "sotis", "args": ["mcp"] }
  }
}
```

### 3. Tell the agent to use it

Add this to your `CLAUDE.md`:

```
At the start of a task, call sotis_start_session with the goal.
After every tool call, call sotis_watch(tool_name, tool_args, result_summary).
If Sotis reports a meltdown, stop, re-read the task, and change approach.
```

### 4. Watch it live

Run `sotis dashboard`, pick the `mcp` session from the selector, and toggle
**Live Mode**.

### Tools exposed

| Tool | Purpose |
|------|---------|
| `sotis_start_session(task_goal)` | Begin a monitoring session |
| `sotis_watch(tool_name, tool_args, result_summary)` | Report a tool call, get a meltdown verdict |
| `sotis_status()` | Current status, steps, resets, entropy |
| `sotis_reset()` | Manually clear the rolling window after a deliberate strategy change |

The server drives the entropy, loop, and density detectors directly, so each
`sotis_watch` reply carries the real entropy reading and the specific reason a
meltdown fired (`TOOL_LOOP`, `ENTROPY_PEAK`, or `EDIT_DENSITY`).

---

## Dashboard

```bash
sotis dashboard
```

The dashboard reads structured JSON telemetry from `logs/session_*.json`:

- **Entropy H(t)** — sliding-window Shannon entropy with the meltdown threshold line
- **Meltdown incidents** — each intercept with its step, reason, and entropy value
- **Subtask progress** — per-subtask status, steps, resets, and GDS weight
- **GDS score** — graceful degradation score across the session
- **Step explorer** — inspect any individual tool call

Toggle **Live Mode** to auto-refresh every 2 seconds while an agent is running.
If a session file goes stale (no new events for 12s) the dashboard surfaces a
**no active agent** banner.
