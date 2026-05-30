"""
sotis.obs.app
=============
Sotis Resilience Dashboard — live telemetry viewer for LLM agent sessions.
"""

from __future__ import annotations

import collections
import glob
import html
import json
import math
import os
import time
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sotis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080C14;
}
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ── Metric cards: left-bar, no rounded border ── */
.m-card {
    background: #0D1320;
    border-left: 3px solid #1E293B;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.m-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(255,255,255,.02) 0%, transparent 60%);
    pointer-events: none;
}
.m-label {
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .14rem;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: .5rem;
}
.m-value {
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
    margin: 0;
    letter-spacing: -.03rem;
}

/* ── Status badge: stark pill ── */
.s-badge {
    display: inline-block;
    padding: .18rem .7rem;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .1rem;
    text-transform: uppercase;
    border-radius: 3px;
}
.s-RUNNING     { background: #1D2D44; color: #60A5FA; }
.s-MELTDOWN    { background: #2D1D0A; color: #FB923C; }
.s-COMPLETED   { background: #0D2B1D; color: #34D399; }
.s-RESUMED     { background: #1F1635; color: #A78BFA; }
.s-HARD_FAILED { background: #2D0F0F; color: #F87171; }
.s-INTERRUPTED { background: #2D0F0F; color: #F87171; }
.s-UNKNOWN     { background: #1A1F2E; color: #64748B; }

/* ── Live indicator ── */
.live-pip {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: .7rem; font-weight: 700; letter-spacing: .1rem;
    color: #00E396; text-transform: uppercase;
}
.live-pip::before {
    content: '';
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #00E396;
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: .2; }
}

/* ── Meltdown feed ── */
.md-item {
    border-left: 3px solid #FF4560;
    background: #130A0A;
    padding: .65rem 1rem;
    margin-bottom: .4rem;
}
.md-step   { font-size: .72rem; color: #FF4560; font-weight: 800;
             letter-spacing: .06rem; font-family: 'JetBrains Mono', monospace; }
.md-detail { font-size: .76rem; color: #4B5563; margin-top: .15rem; }

/* ── Subtask rows ── */
.st-row {
    border-left: 3px solid #1E293B;
    background: #0D1320;
    padding: .6rem 1rem;
    margin-bottom: .35rem;
}
.st-title  { font-size: .84rem; font-weight: 700; }
.st-detail { font-size: .7rem; color: #374151; margin-top: .2rem;
             font-family: 'JetBrains Mono', monospace; }

/* ── Section headings ── */
.section-head {
    font-size: .65rem; font-weight: 800; letter-spacing: .18rem;
    text-transform: uppercase; color: #374151;
    border-bottom: 1px solid #0F1828;
    padding-bottom: .4rem; margin-bottom: 1rem;
}

/* ── Divider ── */
hr { border-color: #0F1828 !important; margin: .8rem 0 1.2rem !important; }
</style>
""")

# ─── Config.toml sync ─────────────────────────────────────────────────────────

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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_session_name(path: str) -> str:
    name = os.path.basename(path).replace("session_", "").replace(".json", "")
    if name.startswith("bench-"):
        parts = name.split("-")
        if len(parts) >= 5:
            agent   = "Sotis" if parts[1] == "sotis" else "Baseline"
            domain  = parts[2].upper()
            horizon = parts[3].replace("_", " ")
            run     = parts[4]
            return f"{agent}  ·  {domain}  ·  {horizon}  ·  run {run}"
    if name.startswith("sotis-lg-"):
        return f"live  ·  {name[len('sotis-lg-'):]}"
    if name.startswith("sotis-mcp-"):
        return f"mcp  ·  {name[len('sotis-mcp-'):]}"
    return name


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


def _load_full(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return _read_json_lines(fh)


def _load_incremental(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(st.session_state.live_file_pos)
        ns, nm, nsn = _read_json_lines(fh)
        return ns, nm, nsn, fh.tell()

# ─── Discover sessions ────────────────────────────────────────────────────────

log_files = sorted(glob.glob(os.path.join("logs", "session_*.json")), reverse=True)

if not log_files:
    st.markdown(
        "<div style='padding:5rem 0;text-align:center'>"
        "<div style='font-size:.65rem;font-weight:800;letter-spacing:.18rem;"
        "text-transform:uppercase;color:#1E293B;margin-bottom:1.5rem'>No Data</div>"
        "<div style='font-size:1.1rem;font-weight:700;color:#1E293B'>No session files in logs/</div>"
        "<div style='font-size:.85rem;color:#0F172A;margin-top:.5rem'>"
        "Run <code>sotis benchmark</code> or connect an agent with SotisLangGraphGuard</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ─── HEADER ROW ───────────────────────────────────────────────────────────────

brand_col, selector_col, toggle_col = st.columns([2, 6, 2])

with brand_col:
    st.markdown(
        "<div style='padding-top:.45rem'>"
        "<span style='font-size:1.45rem;font-weight:900;letter-spacing:-.03rem;"
        "background:linear-gradient(135deg,#00F2FE 0%,#4FACFE 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        "SOTIS</span>"
        "<div style='font-size:.58rem;font-weight:700;letter-spacing:.18rem;"
        "text-transform:uppercase;color:#1E293B;margin-top:.15rem'>"
        "Resilience Dashboard</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with selector_col:
    selected_path = st.selectbox(
        "session",
        log_files,
        format_func=_fmt_session_name,
        label_visibility="collapsed",
    )

with toggle_col:
    st.write("")
    live_mode = st.toggle(
        "Live Mode",
        value=False,
        help="Auto-refreshes every 2s. Run your agent in another terminal.",
    )

st.divider()

# ─── Load data ────────────────────────────────────────────────────────────────

if st.session_state.live_last_file != selected_path:
    st.session_state.live_file_pos  = 0
    st.session_state.live_steps     = []
    st.session_state.live_meltdowns = []
    st.session_state.live_snapshots = []
    st.session_state.live_last_file = selected_path

if live_mode:
    ns, nm, nsn, new_pos = _load_incremental(selected_path)
    st.session_state.live_steps.extend(ns)
    st.session_state.live_meltdowns.extend(nm)
    st.session_state.live_snapshots.extend(nsn)
    st.session_state.live_file_pos = new_pos
    steps           = st.session_state.live_steps
    meltdowns       = st.session_state.live_meltdowns
    state_snapshots = st.session_state.live_snapshots
else:
    steps, meltdowns, state_snapshots = _load_full(selected_path)

last_snapshot = state_snapshots[-1] if state_snapshots else None
status        = last_snapshot.get("status", "RUNNING") if last_snapshot else "RUNNING"
total_resets  = last_snapshot.get("total_resets", 0)   if last_snapshot else len(meltdowns)
step_count    = last_snapshot.get("step_count", len(steps)) if last_snapshot else len(steps)

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
color_resets = "#00E396" if total_resets == 0 else "#FEB019" if total_resets == 1 else "#FF4560"
color_gds    = "#00E396" if gds_val > 0.7 else "#FEB019" if gds_val > 0.4 else "#FF4560"
bar_resets   = "#00E396" if total_resets == 0 else "#FEB019" if total_resets == 1 else "#FF4560"
bar_gds      = "#00E396" if gds_val > 0.7 else "#FEB019" if gds_val > 0.4 else "#FF4560"

live_indicator = "<span class='live-pip'>live</span>" if live_mode else ""

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(
        f"<div class='m-card' style='border-left-color:#4FACFE'>"
        f"<div class='m-label'>Status &nbsp;{live_indicator}</div>"
        f"<div style='margin-top:.4rem'><span class='s-badge s-{badge_key}'>{html.escape(status)}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc2:
    st.markdown(
        f"<div class='m-card' style='border-left-color:#4FACFE'>"
        f"<div class='m-label'>Steps</div>"
        f"<div class='m-value' style='color:#60A5FA'>{step_count}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc3:
    st.markdown(
        f"<div class='m-card' style='border-left-color:{bar_resets}'>"
        f"<div class='m-label'>Resets</div>"
        f"<div class='m-value' style='color:{color_resets}'>{total_resets}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc4:
    st.markdown(
        f"<div class='m-card' style='border-left-color:{bar_gds}'>"
        f"<div class='m-label'>GDS Score</div>"
        f"<div class='m-value' style='color:{color_gds}'>{gds_val:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.write("")

# ─── ENTROPY CHART — full width ───────────────────────────────────────────────

st.markdown("<div class='section-head'>Entropy H(t) — Tool Sequence Diversity</div>",
            unsafe_allow_html=True)

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
    x_axis = alt.X("Step:Q", axis=alt.Axis(
        grid=True, gridColor="#0F1828", gridOpacity=1,
        labelColor="#374151", titleColor="#374151",
        domainColor="#0F1828", tickColor="#0F1828",
        labelFont="JetBrains Mono", titleFont="Inter",
    ))
    y_axis = alt.Y("H(t):Q", scale=alt.Scale(domain=[0, 3.2]), axis=alt.Axis(
        grid=True, gridColor="#0F1828", gridOpacity=1,
        labelColor="#374151", titleColor="#374151",
        domainColor="#0F1828", tickColor="#0F1828",
        labelFont="JetBrains Mono", titleFont="Inter",
    ))

    # Gradient area fill
    gradient_area = alt.Chart(df_e).mark_area(
        line={"color": "#00F2FE", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(0,242,254,0.25)", offset=1),
                alt.GradientStop(color="rgba(0,242,254,0.0)",  offset=0),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(x=x_axis, y=y_axis)

    # Meltdown threshold
    threshold = alt.Chart(pd.DataFrame({"y": [1.5]})).mark_rule(
        color="#FF4560", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y="y:Q")

    # Meltdown dots
    dots = alt.Chart(df_e).mark_point(size=120, filled=True, color="#FEB019").encode(
        x=x_axis, y=y_axis,
        opacity=alt.condition(alt.datum["Meltdown"], alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Step:Q"), alt.Tooltip("H(t):Q", format=".3f")],
    )

    chart = (
        (gradient_area + threshold + dots)
        .properties(height=240, background="#080C14")
        .configure_view(strokeWidth=0, fill="#080C14")
    )
    st.altair_chart(chart, width="stretch")
    st.markdown(
        "<div style='font-size:.65rem;color:#1E293B;letter-spacing:.04rem;margin-top:-.3rem'>"
        "RED DASHED = MELTDOWN THRESHOLD (H = 1.5 bits) &nbsp;·&nbsp; AMBER DOTS = INTERCEPTED MELTDOWNS"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div style='padding:2rem;text-align:center;color:#1E293B;font-size:.85rem'>"
        "No trajectory data. Select a session or enable Live Mode.</div>",
        unsafe_allow_html=True,
    )

st.write("")

# ─── MIDDLE ROW ───────────────────────────────────────────────────────────────

dag_col, feed_col = st.columns(2)

with dag_col:
    st.markdown("<div class='section-head'>Subtask Progress</div>", unsafe_allow_html=True)
    if last_snapshot and last_snapshot.get("subtasks"):
        _colors = {
            "DONE":        "#00E396",
            "ACTIVE":      "#00F2FE",
            "PENDING":     "#1E293B",
            "HARD_FAILED": "#FF4560",
        }
        _icons = {"DONE": "✓", "ACTIVE": "▶", "PENDING": "○", "HARD_FAILED": "✕"}
        for s in last_snapshot["subtasks"]:
            c = _colors.get(s["status"], "#374151")
            i = _icons.get(s["status"], "·")
            st.markdown(
                f"<div class='st-row' style='border-left-color:{c}'>"
                f"<div class='st-title' style='color:{c}'>{i} &nbsp;{html.escape(s['description'])}</div>"
                f"<div class='st-detail'>"
                f"steps {s.get('completed_steps',0)} &nbsp;/&nbsp; "
                f"resets {s.get('resets_used',0)} &nbsp;/&nbsp; "
                f"weight {s['gds_weight']*100:.0f}%"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='font-size:.8rem;color:#1E293B;padding:.5rem 0'>"
            "No subtask decomposition data for this session.</div>",
            unsafe_allow_html=True,
        )

with feed_col:
    st.markdown("<div class='section-head'>Meltdown Incidents</div>", unsafe_allow_html=True)
    if meltdowns:
        with st.container(height=300):
            for m in reversed(meltdowns):
                at     = m.get("triggered_at_step", "?")
                rsn    = m.get("reason", "MELTDOWN")
                ev     = m.get("entropy_value", "")
                lt     = m.get("loop_tool") or ""
                ev_s   = f"H={ev:.3f}" if isinstance(ev, float) else ""
                detail = "  ·  ".join(filter(None, [lt, ev_s])) or "context reset · workspace rolled back"
                st.markdown(
                    f"<div class='md-item'>"
                    f"<div class='md-step'>[STEP {at}] &nbsp; {html.escape(rsn)}</div>"
                    f"<div class='md-detail'>{html.escape(detail)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            "<div style='font-size:.8rem;color:#00E396;padding:.5rem 0'>"
            "● &nbsp;No meltdowns — agent running cleanly.</div>",
            unsafe_allow_html=True,
        )

st.write("")

# ─── STEP EXPLORER ────────────────────────────────────────────────────────────

with st.expander("STEP EXPLORER", expanded=False):
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
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:.8rem;line-height:1.8;color:#374151'>"
                f"<span style='color:#4FACFE'>STEP</span> &nbsp;{active.get('step_index', step_idx)}<br>"
                f"<span style='color:#4FACFE'>TOOL</span> &nbsp;{html.escape(active['tool_name'])}<br>"
                f"{'<span style=color:#4FACFE>NODE</span> &nbsp;' + html.escape(active['node']) if 'node' in active else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with right:
            st.caption("ARGS")
            st.json(active.get("tool_args", {}), expanded=True)
        if active.get("result_summary"):
            st.caption("RESULT")
            st.code(active["result_summary"], language="text")
    else:
        st.markdown(
            "<div style='font-size:.8rem;color:#1E293B;padding:.5rem 0'>No steps yet.</div>",
            unsafe_allow_html=True,
        )

# ─── LIVE AUTO-REFRESH ────────────────────────────────────────────────────────
if live_mode:
    time.sleep(2)
    st.rerun()
