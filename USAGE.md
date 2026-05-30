# Sotis — Usage Guide

CLI commands and the live dashboard.

> Library usage (the `SotisGuard` API) and the LangGraph integration live in the
> [README](README.md). Architecture and design decisions live in
> [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Installation

```bash
pip install sotis

# with the dashboard
pip install sotis[obs]

# with the LangGraph integration
pip install sotis[langgraph]
```

---

## CLI Commands

```bash
sotis dashboard    # Launch the Streamlit observability dashboard
sotis benchmark    # Run the empirical benchmark suite
sotis demo         # Run the built-in meltdown/recovery demo
```

| Command | What it does |
|---------|--------------|
| `sotis dashboard` | Opens the live telemetry dashboard. Reads `logs/session_*.json` and renders entropy, meltdowns, subtask progress, and the GDS score. Toggle **Live Mode** to auto-refresh during an active session. |
| `sotis benchmark` | Runs the empirical benchmark suite and writes session logs the dashboard can replay. |
| `sotis demo` | Runs the built-in meltdown → intercept → recovery demo. |

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

Sessions are produced by `sotis benchmark` or by an agent wrapped with
`SotisLangGraphGuard` (see the [README](README.md)).
