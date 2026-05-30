"""
sotis.obs.app
=============
Sotis Resilience Dashboard — live telemetry viewer for LLM agent sessions.
Supports structured JSON telemetry (live + static) and raw Track 2 audit logs.
"""

from __future__ import annotations

import collections
import glob
import html
import json
import math
import os
import re
import time
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Sotis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
code, pre, .stCode        { font-family: 'JetBrains Mono', monospace !important; }

.metric-card {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color .2s ease;
}
.metric-card:hover { border-color: rgba(0,242,254,0.3); }
.metric-label {
    font-size: .72rem; font-weight: 700; letter-spacing: .09rem;
    text-transform: uppercase; color: #475569; margin-bottom: .45rem;
}
.metric-value { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 0; }

.badge {
    display: inline-block; padding: .22rem .85rem; border-radius: 999px;
    font-size: .78rem; font-weight: 700; letter-spacing: .05rem;
}
.badge-RUNNING     { background:rgba(59,130,246,.12); color:#60A5FA; border:1px solid rgba(59,130,246,.3); }
.badge-MELTDOWN    { background:rgba(245,158,11,.12);  color:#FBBF24; border:1px solid rgba(245,158,11,.3); }
.badge-COMPLETED   { background:rgba(16,185,129,.12);  color:#34D399; border:1px solid rgba(16,185,129,.3); }
.badge-RESUMED     { background:rgba(139,92,246,.12);  color:#A78BFA; border:1px solid rgba(139,92,246,.3); }
.badge-HARD_FAILED { background:rgba(239,68,68,.12);   color:#F87171; border:1px solid rgba(239,68,68,.3); }
.badge-INTERRUPTED { background:rgba(239,68,68,.12);   color:#F87171; border:1px solid rgba(239,68,68,.3); }
.badge-UNKNOWN     { background:rgba(100,116,139,.12); color:#94A3B8; border:1px solid rgba(100,116,139,.3); }

.live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #34D399; margin-right: 5px; vertical-align: middle;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.3; transform:scale(.7); }
}

.meltdown-item {
    background: rgba(239,68,68,.07);
    border: 1px solid rgba(239,68,68,.18);
    border-radius: 8px; padding: .7rem 1rem; margin-bottom: .45rem;
}
.meltdown-step   { font-size: .78rem; color: #F87171; font-weight: 700; }
.meltdown-reason { font-size: .78rem; color: #64748B; margin-top: .15rem; }

.subtask-row {
    padding: .55rem .8rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: .45rem;
    background: rgba(15,23,42,.45);
}
.subtask-title  { font-size: .84rem; font-weight: 600; }
.subtask-detail { font-size: .73rem; color: #475569; margin-top: .18rem; }
</style>
""")

# ─── Session state ────────────────────────────────────────────────────────────
for _k, _v in [
    ("live_file_pos",   0),
    ("live_steps",      []),
    ("live_meltdowns",  []),
    ("live_snapshots",  []),
    ("live_last_file",  ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Sidebar: data source only ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<span style='font-size:1.1rem;font-weight:800;"
        "background:linear-gradient(135deg,#00F2FE,#4FACFE);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        "🛡️ Sotis</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    log_mode = st.radio(
        "Data source",
        ["JSON Telemetry", "Audit Logs (Track 2)"],
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _scan_txt_logs() -> List[str]:
    return sorted(glob.glob(
        os.path.join("ExperimentLog", "**", "run_*.txt"), recursive=True
    ))


def _parse_txt_log(file_path: str) -> dict:
    steps, meltdowns, recoveries = [], [], []
    node, tool, tool_args = "INIT", None, {}
    tool_result: List[str] = []
    collecting = False
    lines: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        for line in lines:
            s = line.strip()
            m = re.match(r"^--- \[Node:\s*([^\]]+)\] ---", s)
            if m:
                if tool:
                    steps.append({"step_index": len(steps)+1, "node": node,
                                  "tool_name": tool, "tool_args": tool_args,
                                  "result_summary": "\n".join(tool_result).strip()})
                    tool, tool_args, tool_result = None, {}, []
                node = m.group(1); collecting = False; continue
            cm = re.match(r"^Assistant calls:\s*([a-zA-Z0-9_]+)\((.*)\)", s)
            if cm:
                tool = cm.group(1)
                try:
                    tool_args = json.loads(cm.group(2).replace("'", '"'))
                except Exception:
                    tool_args = {"raw": cm.group(2)}
                collecting = False; continue
            if s.startswith("Tool Result:"):
                collecting = True
                if s[len("Tool Result:"):].strip():
                    tool_result.append(s[len("Tool Result:"):].strip())
                continue
            if collecting:
                if s.startswith("---") or "Meltdown" in s:
                    collecting = False
                else:
                    tool_result.append(s)
            if "Meltdown intercepted" in line or "[Sotis] Meltdown intercepted!" in line:
                reason = ("EDIT_DENSITY" if "density" in line.lower()
                          else "TOOL_LOOP" if "TOOL_LOOP" in line else "ENTROPY_PEAK")
                meltdowns.append({"triggered_at_step": len(steps)+1, "reason": reason, "msg": s})
            if "rolling back modified files" in line.lower():
                recoveries.append({"step": len(steps)+1, "msg": s})
        if tool:
            steps.append({"step_index": len(steps)+1, "node": node,
                          "tool_name": tool, "tool_args": tool_args,
                          "result_summary": "\n".join(tool_result).strip()})
    except Exception as e:
        st.error(f"Parse error: {e}")
    joined = "".join(lines)
    return {
        "steps": steps, "meltdowns": meltdowns, "recoveries": recoveries,
        "status": "COMPLETED" if "finished in" in joined else "INTERRUPTED",
        "total_resets": len(meltdowns),
    }


def _read_json_lines(fh) -> tuple[List, List, List]:
    steps, meltdowns, snapshots = [], [], []
    for line in fh:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            et, d = rec.get("event_type"), rec.get("data", {})
            if et == "step":       steps.append(d)
            elif et == "meltdown": meltdowns.append(d)
            elif et == "state":    snapshots.append(d)
        except Exception:
            pass
    return steps, meltdowns, snapshots


def _load_json_full(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return _read_json_lines(fh)


def _load_json_incremental(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(st.session_state.live_file_pos)
        ns, nm, nsn = _read_json_lines(fh)
        return ns, nm, nsn, fh.tell()


# ─── Scan available files (before rendering header) ───────────────────────────

if log_mode == "JSON Telemetry":
    available_files = sorted(glob.glob(os.path.join("logs", "session_*.json")), reverse=True)
    fmt_fn = os.path.basename
else:
    available_files = _scan_txt_logs()
    fmt_fn = lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}"

if not available_files:
    st.markdown(
        "<div style='padding:3rem 0;text-align:center;color:#475569'>"
        "<div style='font-size:3rem'>🛡️</div>"
        "<div style='font-size:1.2rem;font-weight:700;margin:.5rem 0'>No session files found</div>"
        "<div style='font-size:.9rem'>Run <code>sotis benchmark</code> or start an agent with <code>SotisLangGraphGuard</code></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ─── HEADER ROW: branding | session selector | live toggle ────────────────────

brand_col, selector_col, toggle_col = st.columns([2, 5, 2])

with brand_col:
    st.markdown(
        "<div style='padding-top:.35rem'>"
        "<span style='font-size:1.6rem;font-weight:800;"
        "background:linear-gradient(135deg,#00F2FE,#4FACFE);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        "🛡️ Sotis</span>"
        "<div style='font-size:.72rem;color:#334155;margin-top:.1rem'>Resilience Dashboard</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with selector_col:
    selected_log_path = st.selectbox(
        "session",
        available_files,
        format_func=fmt_fn,
        label_visibility="collapsed",
    )

with toggle_col:
    st.write("")   # nudge toggle down to align with selectbox
    live_mode = st.toggle(
        "🔴  Live Mode",
        value=False,
        help="Auto-refreshes every 2s. Run your agent in another terminal.",
    )

st.divider()

# ─── Load data ────────────────────────────────────────────────────────────────

steps:           List[dict] = []
meltdowns:       List[dict] = []
state_snapshots: List[dict] = []
recoveries:      List[dict] = []
status        = "RUNNING"
total_resets  = 0
step_count    = 0
last_snapshot = None

if log_mode == "JSON Telemetry":
    if st.session_state.live_last_file != selected_log_path:
        st.session_state.live_file_pos  = 0
        st.session_state.live_steps     = []
        st.session_state.live_meltdowns = []
        st.session_state.live_snapshots = []
        st.session_state.live_last_file = selected_log_path

    if live_mode:
        ns, nm, nsn, new_pos = _load_json_incremental(selected_log_path)
        st.session_state.live_steps.extend(ns)
        st.session_state.live_meltdowns.extend(nm)
        st.session_state.live_snapshots.extend(nsn)
        st.session_state.live_file_pos = new_pos
        steps           = st.session_state.live_steps
        meltdowns       = st.session_state.live_meltdowns
        state_snapshots = st.session_state.live_snapshots
    else:
        steps, meltdowns, state_snapshots = _load_json_full(selected_log_path)

    last_snapshot = state_snapshots[-1] if state_snapshots else None
    status        = last_snapshot.get("status", "RUNNING") if last_snapshot else "RUNNING"
    total_resets  = last_snapshot.get("total_resets", 0)   if last_snapshot else len(meltdowns)
    step_count    = last_snapshot.get("step_count", len(steps)) if last_snapshot else len(steps)

else:
    parsed        = _parse_txt_log(selected_log_path)
    steps         = parsed["steps"]
    meltdowns     = parsed["meltdowns"]
    recoveries    = parsed["recoveries"]
    status        = parsed["status"]
    total_resets  = parsed["total_resets"]
    step_count    = len(steps)

# ─── GDS ──────────────────────────────────────────────────────────────────────
if last_snapshot and last_snapshot.get("subtasks"):
    gds_val = sum(
        s["gds_weight"] * max(0.0, 1.0 - s.get("resets_used", 0) * 0.2)
        for s in last_snapshot["subtasks"] if s["status"] == "DONE"
    )
else:
    gds_val = max(0.0, 1.0 - total_resets * 0.2)

# ─── METRIC ROW ───────────────────────────────────────────────────────────────

badge_key    = status if status in ("RUNNING","MELTDOWN","COMPLETED","RESUMED","HARD_FAILED","INTERRUPTED") else "UNKNOWN"
color_resets = "#34D399" if total_resets == 0 else "#FBBF24" if total_resets == 1 else "#F87171"
color_gds    = "#34D399" if gds_val > 0.7  else "#FBBF24" if gds_val > 0.4  else "#F87171"
live_badge   = "<span class='live-dot'></span>" if live_mode else ""

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Status {live_badge}</div>
        <div style='margin-top:.35rem'><span class='badge badge-{badge_key}'>{html.escape(status)}</span></div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Steps</div>
        <div class='metric-value' style='color:#60A5FA'>{step_count}</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Resets</div>
        <div class='metric-value' style='color:{color_resets}'>{total_resets}</div>
    </div>""", unsafe_allow_html=True)
with mc4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>GDS Score</div>
        <div class='metric-value' style='color:{color_gds}'>{gds_val:.3f}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ─── ENTROPY CHART — full width ───────────────────────────────────────────────

st.markdown("#### Entropy H(t) — Tool Sequence Diversity")

if steps:
    meltdown_steps = {m.get("triggered_at_step", -1) for m in meltdowns}
    entropy_rows: List[dict] = []
    window: List[str] = []

    for idx, step_item in enumerate(steps):
        window.append(step_item["tool_name"])
        if len(window) > 5:
            window.pop(0)
        counts = collections.Counter(window)
        total  = len(window)
        h = -sum((c / total) * math.log2(c / total) for c in counts.values())
        entropy_rows.append({
            "Step":     idx + 1,
            "H(t)":     round(h, 4),
            "Meltdown": (idx + 1) in meltdown_steps,
        })

    df_e   = pd.DataFrame(entropy_rows)
    x_axis = alt.X("Step:Q", axis=alt.Axis(grid=False, labelColor="#475569", titleColor="#475569",
                                            domainColor="#1E293B", tickColor="#1E293B"))
    y_axis = alt.Y("H(t):Q", scale=alt.Scale(domain=[0, 3.2]),
                   axis=alt.Axis(labelColor="#475569", titleColor="#475569",
                                 domainColor="#1E293B", tickColor="#1E293B"))

    area  = alt.Chart(df_e).mark_area(color="#4FACFE", opacity=0.07).encode(x=x_axis, y=y_axis)
    line  = alt.Chart(df_e).mark_line(color="#4FACFE", strokeWidth=2.5).encode(x=x_axis, y=y_axis)
    thres = alt.Chart(pd.DataFrame({"y": [1.5]})).mark_rule(
        color="#F87171", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(y="y:Q")
    dots  = alt.Chart(df_e).mark_point(size=110, filled=True, color="#F59E0B").encode(
        x=x_axis, y=y_axis,
        opacity=alt.condition(alt.datum["Meltdown"], alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Step:Q"), alt.Tooltip("H(t):Q", format=".3f")],
    )

    chart = (
        (area + line + thres + dots)
        .properties(height=250)
        .configure_view(strokeWidth=0, fill="transparent")
        .configure(background="transparent")
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Red dashed line = meltdown threshold H = 1.5 bits  ·  Orange dots = intercepted meltdowns")
else:
    st.info("No trajectory data yet. Start an agent session or select a session file.")

st.write("")

# ─── MIDDLE ROW: Subtask DAG | Meltdown Feed ──────────────────────────────────

dag_col, feed_col = st.columns(2)

with dag_col:
    st.markdown("#### Subtask Progress")
    if last_snapshot and last_snapshot.get("subtasks"):
        _colors = {"DONE": "#34D399", "ACTIVE": "#60A5FA", "PENDING": "#334155", "HARD_FAILED": "#F87171"}
        _icons  = {"DONE": "✅", "ACTIVE": "⚡", "PENDING": "⏳", "HARD_FAILED": "❌"}
        for s in last_snapshot["subtasks"]:
            c = _colors.get(s["status"], "#475569")
            i = _icons.get(s["status"], "·")
            st.markdown(
                f"<div class='subtask-row' style='border-left:3px solid {c}'>"
                f"<div class='subtask-title' style='color:{c}'>{i} {html.escape(s['description'])}</div>"
                f"<div class='subtask-detail'>"
                f"Steps: {s.get('completed_steps',0)} &nbsp;·&nbsp; "
                f"Resets: {s.get('resets_used',0)} &nbsp;·&nbsp; "
                f"Weight: {s['gds_weight']*100:.0f}%"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    elif recoveries:
        st.success(f"{len(recoveries)} checkpoint rollback(s)")
        for rec in recoveries:
            st.markdown(f"🔹 Step {rec['step']}: `{rec['msg']}`")
    else:
        st.info("No subtask data for this session.")

with feed_col:
    st.markdown("#### Meltdown Incidents")
    if meltdowns:
        with st.container(height=300):
            for m in reversed(meltdowns):
                at   = m.get("triggered_at_step", "?")
                rsn  = m.get("reason", "MELTDOWN")
                ev   = m.get("entropy_value", "")
                lt   = m.get("loop_tool") or ""
                ev_s = f" · H={ev:.3f}" if isinstance(ev, float) else ""
                detail = f"{lt}{ev_s}".strip(" ·")
                st.markdown(
                    f"<div class='meltdown-item'>"
                    f"<div class='meltdown-step'>⚡ Step {at} — {html.escape(rsn)}</div>"
                    f"<div class='meltdown-reason'>"
                    f"{html.escape(detail) if detail else 'Meltdown intercepted — context reset and workspace rolled back.'}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.success("No meltdowns detected — agent running cleanly.")

st.write("")

# ─── STEP EXPLORER — bottom ───────────────────────────────────────────────────

with st.expander("🔍  Step Explorer", expanded=False):
    if steps:
        if len(steps) > 1:
            step_idx = st.slider(
                "Step",
                min_value=1,
                max_value=len(steps),
                value=len(steps),
                label_visibility="collapsed",
            )
        else:
            step_idx = 1
        active = steps[step_idx - 1]
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"**Step** &nbsp; `{active.get('step_index', step_idx)}`")
            st.markdown(f"**Tool** &nbsp; `{html.escape(active['tool_name'])}`")
            if "node" in active:
                st.markdown(f"**Node** &nbsp; `{html.escape(active['node'])}`")
        with right:
            st.caption("Arguments")
            st.json(active.get("tool_args", {}), expanded=True)
        if active.get("result_summary"):
            st.caption("Result")
            st.code(active["result_summary"], language="text")
    else:
        st.info("No steps to explore yet.")

# ─── LIVE AUTO-REFRESH (must be last) ────────────────────────────────────────
if live_mode:
    time.sleep(2)
    st.rerun()
