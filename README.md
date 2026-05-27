# Sotis

**Reliability middleware for long-horizon LLM agents.**

Sotis sits between users and LLM agents, monitoring execution in real time, detecting meltdowns via sliding-window Shannon entropy analysis and Workspace Edit Density loop detection, and transparently resetting context to restore forward progress.

**Based on:** _"Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents"_ (arXiv:2603.29231v1, April 2026)

---

## Core Capabilities

| Capability | Description |
|---|---|
| **Meltdown Detection** | Sliding-window Shannon entropy (N=5, H_threshold=1.5) + exact loop detection |
| **Workspace Density Guard** | Detects infinite same-file edit cycles (3 edits without test result changes) |
| **Transparent Reset** | Git-diff checkpointing + distilled context rebuild (>=60% token savings) |
| **Graceful Degradation** | GDS scoring rewards partial progress even if a meltdown occurs |
| **LLM Support** | OpenAI, Anthropic, DeepSeek, Google Gemini via custom ReAct runtime |
| **Observability** | Premium Streamlit dashboard + structured JSON session logs + text log parser |
| **LangGraph Integration** | Native middleware node intercepting state and rolling back files |
| **Document Processing** | Unified multi-format parser (PDF, XLSX, Word, CSV) + Token-based Jaccard similarity (threshold >= 0.65) semantic loop detector |

---

## Project Structure

```
Sotis/
├── sotis/
│   ├── core/         # Pure computation: entropy, loops, checkpoint, decomposition, GDS
│   ├── lib/          # ReAct runtime, LangGraph integration + LLM adapters
│   ├── obs/          # Premium Streamlit dashboard + structured logger
│   └── bench/        # Benchmark harness and task generators
├── tests/            # High-coverage automated test suite (125 tests)
├── ExperimentLog/    # Real-agent Track 2 stress test run logs
├── performance_metrics.txt   # Empirical proof-of-performance ledger
├── pyproject.toml    # Modern package configuration
├── requirements.txt  # Core dependencies
└── pytest.ini        # Testing settings
```

---

## Getting Started

### Installation

```bash
# Clone the repository and navigate inside
cd Sotis

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install Sotis in editable developer mode
pip install -e .
```

### The 3-Line Promise

Use Sotis to secure any LLM agent against infinite loops and disorganised tool calling in exactly three lines of code:

```python
from sotis import SotisGuard

# 1. Initialize the guard
guard = SotisGuard()

# 2. Watch tool invocations in your agent's loop
meltdown_detected = guard.watch(tool_name="write_file", tool_args={"path": "app.py"}, result_summary="Written successfully")

# 3. Intercept and recover if Sotis detects a meltdown
if meltdown_detected:
    print("Sotis blocked an infinite loop! Triggering transparent reset...")
    guard.reset()
```

### Premium Telemetry Dashboard

Launch Sotis's visual Streamlit dashboard to replay structured telemetry logs or parse Track 2 agent stress test logs:

```bash
# Launch the dashboard
streamlit run sotis/obs/app.py
```

---

## Build Phases

All phases of the project have been built, rigorously tested, and successfully finalized:

| Phase | Focus | Status |
|---|---|---|
| **1** | Core Engine: Schemas, Shannon Entropy, Loop Detection | COMPLETED |
| **2** | Checkpoint, Context Reset and Resumption | COMPLETED |
| **3** | Task Decomposition and GDS Scorer | COMPLETED |
| **4** | Telemetry, Streamlit Dashboard and Benchmark Harness | COMPLETED |
| **5** | LangGraph Node Middleware Integration and Packaging | COMPLETED |
| **6** | Workspace Density Guard and AST Query Engine Stress Testing | COMPLETED |
| **7** | Streamlit Dashboard Polish and Interactive Log Viewer | COMPLETED |
| **8** | Three-Line Developer Facade API and End-to-End Validation | COMPLETED |
| **9** | GitHub Release Preparation and Packaging Metadata | COMPLETED |
| **10** | Live Web Research Stress Test and Crawler Tools | COMPLETED |
| **11** | Document Handling and Jaccard Semantic Loops (PDF, Excel, Word, CSV) | COMPLETED |

---

## Performance Metrics

See [`performance_metrics.txt`](./performance_metrics.txt) — a real-time updated ledger tracking empirical outcomes across all phases showing:
- **Entropy and Loop Detection Latency**: < 0.2ms overhead per step.
- **Context Distillation Reduction**: **86.14%** token savings in the resumption briefing.
- **Resilience Gains**: Real-world recovery on circular imports and AST recursive loop traps.
