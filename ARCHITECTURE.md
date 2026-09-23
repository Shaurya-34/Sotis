# Sotis: Active Runtime Stabilizer for Long-Horizon LLM Agents
🛡️ *Watches your LLM agent and catches it before it spirals.*

Sotis is a high-performance, low-latency reliability middleware framework designed for long-horizon agent execution. It operationalizes the formal reliability framework proposed in the paper **["Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents"](https://arxiv.org/abs/2603.29231)** (arXiv:2603.29231, April 2026) to detect behavioral meltdowns in real-time, roll back corrupting workspace edits, and transparently reset agent context for a clean resumption.

---

## 🏗️ Architectural Overview

Sotis operates as a single-threaded runtime interceptor layer situated between the LLM Reasoning Engine (raw or ReAct/LangGraph workflow) and the Target Tool Workspace. It is organized into three distinct layers:

```mermaid
graph TD
    %% Styling and layout
    classDef default fill:#1E293B,stroke:#334155,color:#F8FAFC;
    classDef core fill:#0F172A,stroke:#00F2FE,color:#00F2FE,stroke-width:2px;
    classDef lib fill:#0F172A,stroke:#3B82F6,color:#3B82F6,stroke-width:2px;
    classDef obs fill:#0F172A,stroke:#10B981,color:#10B981,stroke-width:2px;

    %% Nodes
    A[Agent Goal / User Prompt] --> B(Task Decomposer)
    B --> C[Topological DAG of Subtasks]
    C --> D[Sotis ReAct Runtime / LangGraph Guard]
    
    subgraph Core ["Sotis Core Layer (Pure Python & Math)"]
        E(Shannon Entropy Monitor)
        F(Jaccard & Fingerprint Loop Detector)
        G(Workspace Density Guard)
        H(Checkpoint Manager)
        I(Context Resetter)
        J(GDS Scorer)
    end
    class E,F,G,H,I,J core;

    subgraph Adapters ["LLM Adapters"]
        K[OpenAI Adapter]
        L[Anthropic Adapter]
        M[DeepSeek Adapter]
    end
    class K,L,M lib;

    D --> E & F & G
    E & F & G -->|Meltdown Intercept| H
    H -->|Workspace Rollback| TargetWorkspace[("Target File Workspace")]
    H -->|Unified Diffs| I
    I -->|Distilled Resumption Prompt| D
    
    subgraph Observability ["Observability & Telemetry"]
        N[JSON-L Logger]
        O[Streamlit Dashboard]
    end
    class N,O obs;
    
    D -->|Step Telemetry| N
    N --> O
```

---

## 🛡️ Core Capabilities & Features

### 1. Multi-Faceted Meltdown Detection
Sotis uses three complementary, lightweight monitors in the hot-path (averaging **< 0.02ms** execution latency, 100x faster than the 2ms SLA target) to detect execution collapse:

*   **Sliding-Window Shannon Entropy**: Computes $H(t) = -\sum p(x) \log_2 p(x)$ of tool calls over a rolling 5-step window. 
    *   *Hard Threshold*: Triggered immediately if $H(t) \ge 1.5$ bits (indicating highly chaotic, disorganized tool switching).
    *   *Entropy Trend Warning*: Early warning triggered by 3 consecutive steps of strictly increasing entropy, signaling gradual strategy degradation before a total blowout.
*   **Exact and Jaccard Semantic Loop Detector**:
    *   *Exact Matches*: Flags if the exact same `(tool_name, args_hash)` fingerprint is called $\ge 3$ times within a 6-step window (catches tight loops, e.g., reading the same file repeatedly).
    *   *Jaccard Token Similarity*: Extracts query/search strings from tool parameters and computes pairwise Jaccard similarity. If consecutive query similarity $\ge 0.65$ for $\ge 3$ steps, Sotis flags a **Semantic Spiral** (where the agent keeps slightly rephrasing the same query hoping for different search outcomes).
*   **Workspace Density Guard**:
    *   Interceptors uncoordinated, hyperactive file modification storms (a common failure mode where an agent tries to edit a single file repeatedly without moving past a compilation or test block).
    *   Tracks consecutive edits to the same file path. If a file is modified $\ge 3$ times without changing the test suite result summary, a meltdown is declared. A shift in test output cleanly resets the counters.

### 2. Lightweight Git-Style Checkpointing & Rollbacks
*   **Unified Diffs**: To avoid the heavy overhead of full directory snapshots, Sotis uses Python's native `difflib.unified_diff` (zero external Git dependencies, average latency **~1.23ms**) to snapshot files tracked at the beginning of each subtask.
*   **Baselines & Rollbacks**: If a meltdown occurs, Sotis automatically rolls back all tracked files to their last known stable subtask baseline. This prevents the agent from inheriting syntactically broken files or circular dependency traps upon resumption.

### 3. Context Distillation (Resumption Prompting)
*   **Up to ~86% Token Reduction**: On live agent runs of 8–16 steps, the distilled briefing is **67–86%** smaller than the logged history (measured with OpenAI's `cl100k_base` BPE tokenizer via `tiktoken`). The briefing has a fixed cost of ~250–440 tokens, so the saving grows with trajectory length and is negative on very short runs (under ~5 steps). Sotis completely prunes the message history (using LangGraph `RemoveMessage` signals) and replaces it with a clean, structured briefing.
*   **Resumption System Prompt**: Synthesizes a fresh, hyper-compact resumption briefing that injects:
    1.  A friendly notification of why the reset occurred.
    2.  The original high-level task goal.
    3.  A **Verified Progress Checklist** (listing completed subtasks).
    4.  The current active subtask goal, remaining step budget, and reset counters.
    5.  The last $N$ unique tool-call observations (preserving semantic memory).
    6.  A **Workspace State Summary** containing Git-style diff snippets showing recent file changes.

### 4. Graceful Degradation Score (GDS)
Rather than scoring task completion as a binary pass/fail, Sotis measures partial progress using a topologically validated Directed Acyclic Graph (DAG) of subtasks:
$$\text{GDS} = \sum (\text{Subtask Weight} \times \text{Success Multiplier})$$
Where $\text{Success Multiplier} = \max(0.0, 1.0 - (\text{Resets Used} \times 0.2))$ for successfully completed subtasks. In `SotisRuntime`, a subtask that needs more than **2 resets** is marked as `HARD_FAILED` and execution stops. `SotisLangGraphGuard` counts resets for the whole session instead, and hard-fails once it passes `max_resets` (default 5).

---

## 📂 Project Structure & Key Modules

Sotis is structured cleanly to decouple mathematical logic from external frameworks:

```
f:\Sotis/
├── pyproject.toml              # Build dependencies (numpy, pydantic, openai, anthropic, streamlit)
├── requirements.txt            # Package list for deployment
├── README.md                   # Quickstart, Science description, and overview
├── performance_metrics.txt     # Scientific ledger recording empirical gains
│
├── sotis/                      # Primary Package Directory
│   ├── __init__.py             # SotisGuard developer facade class
│   │
│   ├── core/                   # Pure Python / Math Computation Layer (Deterministic, Zero LLM APIs)
│   │   ├── schemas.py          # canonical frozen Pydantic models (StepEvent, MeltdownSignal, Subtask)
│   │   ├── entropy.py          # Sliding-window Shannon Entropy tracker & Trend monitor
│   │   ├── loops.py            # Fingerprint & Jaccard semantic loops + WorkspaceDensityGuard
│   │   ├── checkpoint.py       # Incremental unified-diff baseline and rollback manager
│   │   ├── reset.py            # Context distiller translating trajectories into resumption prompts
│   │   ├── decomposition.py    # Topological cycle-checking and domain-based task parser (SE, WR, DP)
│   │   └── gds.py              # Graceful Degradation Score (GDS) arithmetic
│   │
│   ├── lib/                    # Active Middleware & Integration Layer
│   │   ├── adapters.py         # Unified wrapper adapters for OpenAI, Anthropic, DeepSeek, and Mock LLMs
│   │   ├── runtime.py          # Custom ReAct runtime loop (Observe -> Think -> Act)
│   │   └── langgraph_integration.py  # SotisLangGraphGuard middleware node for native LangGraph graphs
│   │
│   ├── obs/                    # Observability & Diagnostics UI
│   │   ├── app.py              # Premium Streamlit dashboard displaying active state and metrics
│   │   └── logger.py           # Real-time structured JSON-Line session telemetry recorder
│   │
│   └── bench/                  # Empirical Benchmarking Harness
│       ├── runner.py           # scientific comparison runner (Baseline vs Sotis k=3)
│       └── tasks.py            # domain-aware task generators (short, med, long, very_long)
│
└── tests/                      # Verification suite (163 unit tests passing)
    ├── run_live_document_handling.py  # PDF/XLSX vector haystack and corruption stress test
    ├── run_live_langgraph_evaluation.py# Full LangGraph graph stress testing
    ├── run_live_web_research.py    # Offline/online web-research simulation
    └── test_core_entropy.py        # Entropy mathematical correctness tests
```

---

## 📊 Validation: what's measured vs. what's simulated

> **Read this first.** Sotis's evidence comes in two tiers. Be precise about which
> is which — conflating them is the easiest way to lose a reader's trust.

### Tier 1 — Real LLM agents (the evidence that counts)

Detection + verified-good rollback have been validated on **real models** (Groq
Llama-3.3-70B, OpenRouter Gemini) against real failure traps. Full unedited
transcripts in [`ExperimentLog/`](ExperimentLog/). Representative live result:
on a circular-import trap, the agent spiraled into an identical-tool loop; Sotis
detected it (`TOOL_LOOP` @ step 23) and rolled back to a **verified-good
checkpoint** (a state proven to still parse), not the most-recent (possibly
broken) snapshot.

A 6-scenario detection gauntlet scored **100% true-positive detection** with a
**33% false-positive rate at the default threshold** (eliminated by raising it —
this is what motivated the adaptive threshold).

**Honest scope:** these runs prove Sotis *detects real meltdowns and restores a
safe, resumable state*. They do **not** yet prove it raises end-to-end task
*success* — one clean run ended on a rate limit without the weak model passing
the tests. As the README says: Sotis bounds failure; it doesn't guarantee
success. A controlled real-agent A/B (with vs. without Sotis, pass@1 across
seeds) is the next step (`bench/real_runner.py`).

### Tier 2 — Simulation harness (logic sanity-check, NOT a benchmark)

`sotis/bench/runner.py` runs a **scripted** agent — diverse tools while healthy,
then a hardcoded flip into an identical-tool loop at a fixed step. The baseline
arm is **fail-by-construction**; the Sotis arm always recovers. So the
oft-quoted "Baseline 0% → Sotis 100%" and "GDS 0.10 → 0.96" are **guaranteed by
the script, not measured on any model.** Their only legitimate use is verifying
that Sotis's detection/reset/GDS *logic* fires correctly under a controlled
meltdown — an integration test, not a performance claim. Treat those figures
accordingly and do not cite them as agent results.

```bash
python -m sotis.bench.runner   # runs the SIMULATION (logic check), writes logs/ + ledger
```

---

## 🛡️ Sotis Controller: Telemetry Dashboard
Sotis comes packaged with a premium Streamlit web app (`sotis/obs/app.py`) designed to visualize active execution sessions:

1.  **Metric Cards Grid**: Live status badges (`RUNNING`, `MELTDOWN`, `COMPLETED`, `HARD_FAILED`), Total Restarts, Telemetry step counts, and the computed GDS score.
2.  **Entropy Curves H(t)**: A real-time line chart plotting Shannon entropy values step-by-step, making MOP (Meltdown-Onset Point) thresholds visually apparent.
3.  **Topological DAG Checklist**: Displays each subtask in the graph, showing its completion status, consumed steps, and active reset counts.
4.  **Incident Ledger**: Highlights precise step indices where meltdowns were intercepted and files rolled back.
5.  **Interactive Trajectory Explorer**: Allows developers to slide through step history, inspect raw JSON tool parameters, and view outputs side-by-side.
