"""
sotis.obs.app
=============
Premium, highly-visual Streamlit dashboard for real-time and post-hoc Sotis telemetry monitoring.
Supports both structured JSON session logs and raw Track 2 stress test audit logs.
"""

from __future__ import annotations

import glob
import html
import json
import math
import os
import re
import collections
from typing import List
import streamlit as st

# Configure premium dashboard layout
st.set_page_config(
    page_title="Sotis Resilience Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Harmony Fonts and Glassmorphism Stylesheet
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Typography & Font Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    code, pre, [class*="stCode"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main Premium Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    
    /* Subtitle Styling */
    .subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.3);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    
    /* Custom status badges */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-running { background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-meltdown { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-completed { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-failed { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* Timeline styling */
    .timeline-item {
        border-left: 2px solid rgba(255, 255, 255, 0.1);
        padding-left: 1.5rem;
        margin-left: 0.5rem;
        padding-bottom: 1.5rem;
        position: relative;
    }
    .timeline-dot {
        position: absolute;
        left: -6px;
        top: 4px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #4FACFE;
        border: 2px solid #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<div style='text-align: center; padding: 1rem 0;'><h2 style='margin:0; font-weight:800; background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🛡️ Sotis Controller</h2><span style='color: #64748B; font-size: 0.85rem;'>Agent Resilience System</span></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS & PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def scan_text_logs() -> list[str]:
    """Find recursively all run_*.txt files inside ExperimentLog/."""
    files = glob.glob(os.path.join("ExperimentLog", "**", "run_*.txt"), recursive=True)
    return sorted(files)

def parse_txt_stress_log(file_path: str) -> dict:
    """Parses Track 2 stdout/stderr files and distills sequence steps."""
    steps = []
    meltdown_events = []
    recovery_events = []
    
    current_node = "INIT"
    current_tool = None
    current_tool_args = {}
    current_tool_result = []
    collecting_result = False
    lines: List[str] = []
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        for line in lines:
            line_str = line.strip()
            
            # Identify Nodes
            node_match = re.match(r"^--- \[Node:\s*([^\]]+)\] ---", line_str)
            if node_match:
                # Flush previous step if any tool was called
                if current_tool:
                    steps.append({
                        "step_index": len(steps) + 1,
                        "node": current_node,
                        "tool_name": current_tool,
                        "tool_args": current_tool_args,
                        "result_summary": "\n".join(current_tool_result).strip(),
                    })
                    current_tool = None
                    current_tool_args = {}
                    current_tool_result = []
                
                current_node = node_match.group(1)
                collecting_result = False
                continue
            
            # Parse Assistant calls
            call_match = re.match(r"^Assistant calls:\s*([a-zA-Z0-9_]+)\((.*)\)", line_str)
            if call_match:
                current_tool = call_match.group(1)
                args_raw = call_match.group(2)
                try:
                    # Clean up single quotes to parse as JSON
                    args_json = args_raw.replace("'", '"')
                    current_tool_args = json.loads(args_json)
                except Exception:
                    current_tool_args = {"raw": args_raw}
                collecting_result = False
                continue
                
            # Parse Tool results
            if line_str.startswith("Tool Result:"):
                collecting_result = True
                res_content = line_str[len("Tool Result:"):].strip()
                if res_content:
                    current_tool_result.append(res_content)
                continue
                
            if collecting_result:
                if line_str.startswith("---") or "Meltdown" in line_str:
                    collecting_result = False
                else:
                    current_tool_result.append(line_str)
            
            # Detect Meltdown events
            if "Meltdown intercepted" in line or "[Sotis] Meltdown intercepted!" in line:
                reason = "TOOL_LOOP" if "TOOL_LOOP" in line else "ENTROPY_PEAK"
                if "density" in line.lower():
                    reason = "EDIT_DENSITY"
                meltdown_events.append({
                    "triggered_at_step": len(steps) + 1,
                    "reason": reason,
                    "msg": line_str
                })
                
            # Detect rollback / recovery notices
            if "rolling back modified files" in line.lower() or "Rolling back modified files" in line:
                recovery_events.append({
                    "step": len(steps) + 1,
                    "msg": line_str
                })
                
        # Flush final step
        if current_tool:
            steps.append({
                "step_index": len(steps) + 1,
                "node": current_node,
                "tool_name": current_tool,
                "tool_args": current_tool_args,
                "result_summary": "\n".join(current_tool_result).strip(),
            })
    except Exception as e:
        st.sidebar.error(f"Error parsing log file: {e}")
        
    return {
        "steps": steps,
        "meltdowns": meltdown_events,
        "recoveries": recovery_events,
        "status": "COMPLETED" if "finished in" in "".join(lines) else "INTERRUPTED",
        "total_resets": len(meltdown_events)
    }

# ─────────────────────────────────────────────────────────────────────────────
# VIEW CONTROLLER SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

log_mode = st.sidebar.radio(
    "Data Source Mode",
    options=["Structured JSON Telemetry", "Raw Track 2 Audit Logs"],
    index=0
)

steps = []
meltdowns = []
state_snapshots = []
intercepts = []
recoveries = []
status = "RUNNING"
total_resets = 0
last_snapshot = None

if log_mode == "Structured JSON Telemetry":
    log_dir = "logs"
    log_files = sorted(glob.glob(os.path.join(log_dir, "session_*.json")), reverse=True)
    
    if not log_files:
        st.title("🛡️ Sotis Resilience Dashboard")
        st.warning("No structured JSON session files found under `logs/` directory.")
        st.info("💡 Run the benchmark to generate telemetry logs: `python -m sotis.bench.runner`")
        st.stop()
        
    selected_log_path = st.sidebar.selectbox(
        "Select Telemetry File",
        options=log_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    with open(selected_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                event_type = record.get("event_type")
                data = record.get("data", {})
                if event_type == "step":
                    steps.append(data)
                elif event_type == "meltdown":
                    meltdowns.append(data)
                elif event_type == "state":
                    state_snapshots.append(data)
                elif event_type == "meltdown_intercepted":
                    intercepts.append(data)
            except Exception:
                pass
                
    last_snapshot = state_snapshots[-1] if state_snapshots else None
    status = last_snapshot.get("status", "UNKNOWN") if last_snapshot else "RUNNING"
    total_resets = last_snapshot.get("total_resets", 0) if last_snapshot else len(meltdowns)
    step_count = last_snapshot.get("step_count", len(steps)) if last_snapshot else len(steps)

else: # Raw Track 2 Audit Logs
    txt_files = scan_text_logs()
    
    if not txt_files:
        st.title("🛡️ Sotis Resilience Dashboard")
        st.warning("No Track 2 run_*.txt files found inside the `ExperimentLog/` directory hierarchy.")
        st.stop()
        
    selected_log_path = st.sidebar.selectbox(
        "Select Audit Log File",
        options=txt_files,
        format_func=lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x))
    )
    
    parsed = parse_txt_stress_log(selected_log_path)
    steps = parsed["steps"]
    meltdowns = parsed["meltdowns"]
    recoveries = parsed["recoveries"]
    status = parsed["status"]
    total_resets = parsed["total_resets"]
    step_count = len(steps)

# ─────────────────────────────────────────────────────────────────────────────
# RENDERING HERO AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div class='main-header'>Sotis Resilience Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Telemetry, Edit Density Loops & Graceful Degradation Diagnostics (arXiv:2603.29231v1)</div>", unsafe_allow_html=True)

# Calculate live GDS Score based on resets
gds_val = 1.0
if log_mode == "Structured JSON Telemetry" and last_snapshot and last_snapshot.get("subtasks"):
    subtasks_list = last_snapshot["subtasks"]
    gds_val = 0.0
    for s_item in subtasks_list:
        if s_item["status"] == "DONE":
            mult = max(0.0, 1.0 - (s_item.get("resets_used", 0) * 0.2))
            gds_val += s_item["gds_weight"] * mult
else:
    # 0.2 penalty per reset to GDS
    gds_val = max(0.0, 1.0 - (total_resets * 0.2))

# Status Badge Color Setup
badge_css = {
    "RUNNING": "badge-running",
    "MELTDOWN": "badge-meltdown",
    "COMPLETED": "badge-completed",
    "HARD_FAILED": "badge-failed",
    "INTERRUPTED": "badge-failed"
}
badge_class = badge_css.get(status, "badge-running")

# Render 4 Premium Grid Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Execution Status</div>
            <div style='margin-top: 0.5rem;'><span class='badge {badge_class}'>{html.escape(str(status))}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Total Restarts</div>
            <div class='metric-value' style='color: #FBBF24;'>{total_resets}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Telemetry Steps</div>
            <div class='metric-value' style='color: #60A5FA;'>{step_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    color_gds = "#34D399" if gds_val > 0.7 else "#FBBF24" if gds_val > 0.4 else "#F87171"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Graceful Degradation (GDS)</div>
            <div class='metric-value' style='color: {color_gds};'>{gds_val:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PLOT AND CHECKSUMS
# ─────────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2])

with left_col:
    # 1. Trajectory Shannon Entropy Plot
    st.subheader("📊 Tool Sequence Entropy Curve H(t)")
    
    entropy_history = []
    rolling_window = []
    
    for idx, step_item in enumerate(steps):
        rolling_window.append(step_item["tool_name"])
        if len(rolling_window) > 5:
            rolling_window.pop(0)
            
        # Shannon Entropy Math
        counts = collections.Counter(rolling_window)
        total_items = len(rolling_window)
        entropy = 0.0
        for item_count in counts.values():
            p = item_count / total_items
            entropy -= p * math.log2(p)
        entropy_history.append({"Step": idx + 1, "Entropy H(t)": round(entropy, 4)})
        
    if entropy_history:
        st.line_chart(entropy_history, x="Step", y="Entropy H(t)", height=300)
    else:
        st.info("No tool trajectory telemetry sequence available.")
        
    # 2. Checkpoint Files / Rollbacks
    st.subheader("🛡️ Resumption & Integrity Checks")
    if log_mode == "Structured JSON Telemetry" and last_snapshot and last_snapshot.get("subtasks"):
        st.markdown("**Subtasks Decomposition Graph & GDS Checklist**")
        subtasks_list = last_snapshot["subtasks"]
        
        table_lines = [
            "| Subtask ID | Description | Status | Steps | Resets | Weight |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |"
        ]
        status_icons = {"DONE": "✅ DONE", "ACTIVE": "⚡ ACTIVE", "FAILED": "❌ FAILED", "PENDING": "⏳ PENDING"}
        for s_item in subtasks_list:
            st_id = s_item["subtask_id"]
            desc = s_item["description"]
            st_status = status_icons.get(s_item["status"], s_item["status"])
            steps_consumed = s_item.get("completed_steps", 0)
            res_used = s_item.get("resets_used", 0)
            w_percentage = f"{s_item['gds_weight'] * 100:.0f}%"
            table_lines.append(f"| `{st_id}` | {desc} | **{st_status}** | {steps_consumed} | {res_used} | {w_percentage} |")
        st.markdown("\n".join(table_lines))
    else:
        # For raw logs, we parse and render the rollbacks explicitly
        if recoveries:
            st.success(f"**Discovered {len(recoveries)} Checkpoint Rollbacks**")
            for rec in recoveries:
                st.markdown(f"🔹 **Step {rec['step']}**: `{rec['msg']}`")
        else:
            st.info("No workspace checkpoint rollbacks were parsed for this run.")

with right_col:
    # 3. Interception Log
    st.subheader("🚨 Meltdown Interceptions Log")
    if meltdowns:
        for idx, m in enumerate(meltdowns):
            at_step = m.get("triggered_at_step", "?")
            reason = m.get("reason", "MELTDOWN")
            msg = m.get("msg") or m.get("message") or f"Meltdown intercepted! Reason: {reason}"
            st.error(f"**Step {at_step}**: {msg}")
    else:
        st.info("No meltdown incidents were intercepted in this session. Running safely.")
        
    # 4. Step Explorer
    st.subheader("⏱️ Telemetry Sequence Explorer")
    if steps:
        if len(steps) > 1:
            step_idx = st.slider("Select Trajectory Step", min_value=1, max_value=len(steps), value=len(steps))
        else:
            step_idx = 1
        active_step = steps[step_idx - 1]
        
        st.markdown(f"**Step Index**: `{active_step['step_index']}`")
        if "node" in active_step:
            st.markdown(f"**Active Node**: `{active_step['node']}`")
        st.markdown(f"**Tool Invocations**: `{active_step['tool_name']}`")
        st.json(active_step["tool_args"])
        if active_step.get("result_summary"):
            st.markdown(f"**Tool Output Summary**:")
            st.code(active_step["result_summary"], language="text")
    else:
        st.info("No telemetry trajectory steps available.")
