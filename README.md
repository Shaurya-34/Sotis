# Sotis

Sotis is a Python reliability library for long-running LLM agents that detects meltdown behavior and automatically resets execution before agents spiral into loops or context collapse.

---

## The Long-Horizon Failure Problem

Current AI agents fail predictably under long-horizon execution. As they run for longer periods, they accumulate error and drift into terminal failure modes:

* **Infinite Loops**: Repeating the same tool calls with identical arguments.
* **Semantic Spirals**: Rephrasing failed queries or tool calls hoping for different outcomes.
* **Context Poisoning**: Flooding the chat history with massive error traces and linter warnings.
* **Edit Storms**: Making rapid, uncoordinated file edits without shifting validation outputs.

Frontier models do not fail because they are simple, but because long-horizon execution decays their reliability envelope until strategy collapse occurs. Sotis acts as an active runtime stabilizer, monitoring execution in real time, detecting behavioral meltdowns, and transparently resetting context to restore forward progress.

---

## Active Execution Recovery in 3 sets of Lines

Unlike passive tracing systems, Sotis operates actively. It intercepts spiraling tool calls, restores files to the last stable checkpoint, compresses history, and injects a resumption briefing.

```python
from sotis import SotisGuard

# 1. Initialize the reliability guard
guard = SotisGuard()

# 2. Watch tool execution events in your agent's loop
for step in range(max_steps):
    action = agent.decide()
    result = tools.execute(action)
    
    meltdown = guard.watch(action.name, action.args, result.summary)
    
    # 3. Intercept and recover automatically
    if meltdown:
        print("Sotis blocked an execution loop! Restoring stable baseline...")
        guard.reset()
```

### Raw Telemetry Trace

Here is how Sotis stabilizes execution in real time:

```text
[Step 22] write_file -> {"path": "src/main.py", "content": "import math"} | Outcome: SUCCESS
[Step 23] run_tests  -> {"cmd": "pytest"} | Outcome: FAIL (ImportError)
[Step 24] write_file -> {"path": "src/main.py", "content": "import math"} | Outcome: SUCCESS
[Step 25] run_tests  -> {"cmd": "pytest"} | Outcome: FAIL (ImportError)

[WARNING] Anomaly detected: Workspace edit storm and exact argument loops
[INTERCEPT] Sotis Meltdown Interception Triggered!
[RECOVER] Restored workspace files to stable baseline step 22 diff
[RECOVER] Distilled session context history (78% token savings)
[RESUME] Injecting resumption briefing prompt into agent context...

[Step 26] grep_search -> {"query": "math"} | Execution resumed cleanly
```

---

## Visual Execution Flow

```
     Agent Begins Task
             |
             v
   Execution Step Loop
             |
             v
   Entropy Rises / Loops Detected (Step 20)
             |
             v
   Sotis Intercepts Meltdown (Step 25)
             |
             v
   Workspace Rolled Back and Context Distilled (>= 60% Token Savings)
             |
             v
   Resumption Briefing Injected
             |
             v
   Execution Resumes Cleanly to Completion
```

---

## The Science: Research-Inspired Reliability

Sotis operationalizes the formal reliability engineering framework introduced in the April 2026 paper:
*["Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents"](https://arxiv.org/abs/2603.29231)* (arXiv:2603.29231)

We translate key theoretical concepts from this paper directly into active runtime layers:

* **Meltdown Onset Point (MOP)**: Quantifies the transition from coherent planning to chaotic looping. Sotis monitors the Shannon entropy of the tool-call distribution over a sliding window (size w=5). A spike in entropy or a tight repeating sequence triggers an immediate Meltdown Onset Point flag.
* **Reliability Decay**: Demonstrates that agent success rates decay super-linearly with task duration due to positively correlated errors (a confused agent stays confused). Sotis acts as a runtime circuit breaker to reset the error correlation coefficient.
* **Episodic Memory Failures**: The paper proves that naive episodic memory scaffolds universally degrade long-horizon performance by accumulating step budget and context overhead. Sotis maintains reliability through **controlled checkpointed resets** rather than continuous memory accumulation.
* **Graceful Degradation Score (GDS)**: Evaluates agent trajectories using weighted directed acyclic graphs of subtask completion. When a meltdown is intercepted, Sotis scores partial progress and resets execution context, allowing agents to preserve partial GDS while starting fresh.

---

## Strongest Differentiator: Active Stabilization vs Passive Tracing

Passive AI developer tooling (such as LangSmith, Langfuse, or Helicone) merely monitors failure, logging tokens and rendering traces *after* your agent has already spent $20 looping in production. 

Sotis is closer to **runtime reliability middleware**. It does not merely observe agent degradation—it actively intercepts loops, rolls back uncommitted file edits, distills conversation context, and redirects the model's reasoning loop in real time.

---

## Project Structure

```
Sotis/
├── sotis/
│   ├── core/         # Pure computation: entropy, loops, checkpoint, decomposition, GDS
│   ├── lib/          # ReAct runtime, LangGraph integration + LLM adapters
│   ├── obs/          # Telemetry app + structured JSON logger
│   └── bench/        # Benchmark harness and task generators
├── tests/            # High-coverage automated test suite (127 tests)
├── ExperimentLog/    # Real-agent Track 2 stress test run logs
├── performance_metrics.txt   # Empirical proof-of-performance ledger
├── pyproject.toml    # Package configuration
├── requirements.txt  # Core dependencies
└── pytest.ini        # Testing settings
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
