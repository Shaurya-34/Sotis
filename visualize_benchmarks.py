"""
Sotis Benchmark Visualizations
Generates clean, LinkedIn-ready charts from verified performance_metrics.txt data.
Run: python visualize_benchmarks.py
Outputs PNG files to ./charts/
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("charts", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0F172A"
SOTIS    = "#00F2FE"
BASELINE = "#EF4444"
TEXT     = "#F8FAFC"
GRID     = "#1E293B"
ACCENT   = "#3B82F6"

def apply_base_style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)


# ── Chart 1: pass@1 Baseline vs Sotis across horizons (all 3 domains) ────────
horizons = ["Short", "Medium", "Long", "Very Long"]
baseline_pass = [0.0, 0.0, 0.0, 0.0]
sotis_pass    = [100.0, 100.0, 100.0, 100.0]

x = np.arange(len(horizons))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
apply_base_style(fig, ax)

bars_b = ax.bar(x - w/2, baseline_pass, w, color=BASELINE, label="Baseline Agent", zorder=3)
bars_s = ax.bar(x + w/2, sotis_pass,    w, color=SOTIS,    label="Sotis-wrapped Agent", zorder=3)

for bar in bars_s:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            "100%", ha="center", va="bottom", color=SOTIS, fontsize=10, fontweight="bold")
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, 2,
            "0%", ha="center", va="bottom", color=BASELINE, fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(horizons, fontsize=12)
ax.set_ylabel("pass@1 Success Rate (%)", fontsize=12)
ax.set_ylim(0, 118)
ax.set_title("Agent Success Rate: Baseline vs Sotis\n(SE · Web Research · Document Processing)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=GRID, labelcolor=TEXT, fontsize=11, framealpha=0.9)

fig.tight_layout()
fig.savefig("charts/1_pass_at_1_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/1_pass_at_1_comparison.png")


# ── Chart 2: GDS Scores — Baseline vs Sotis (Very Long tasks) ─────────────────
domains = ["Software\nEngineering", "Web\nResearch", "Document\nProcessing"]
gds_baseline = [0.10, 0.10, 0.10]
gds_sotis    = [0.96, 0.96, 0.96]

x = np.arange(len(domains))

fig, ax = plt.subplots(figsize=(8, 5))
apply_base_style(fig, ax)

bars_b = ax.bar(x - w/2, gds_baseline, w, color=BASELINE, label="Baseline Agent", zorder=3)
bars_s = ax.bar(x + w/2, gds_sotis,    w, color=SOTIS,    label="Sotis-wrapped Agent", zorder=3)

for bar, val in zip(bars_s, gds_sotis):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.2f}", ha="center", va="bottom", color=SOTIS, fontsize=11, fontweight="bold")
for bar, val in zip(bars_b, gds_baseline):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.2f}", ha="center", va="bottom", color=BASELINE, fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=12)
ax.set_ylabel("Graceful Degradation Score (GDS)", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Graceful Degradation Score — Very Long Tasks\n(1.0 = perfect completion, partial credit for resets)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=GRID, labelcolor=TEXT, fontsize=11, framealpha=0.9)

fig.tight_layout()
fig.savefig("charts/2_gds_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/2_gds_comparison.png")


# ── Chart 3: Reliability drop from paper (external data, Khanal et al. 2026) ──
task_lengths = ["Short", "Medium", "Long", "Very Long"]
reliability  = [0.90, 0.72, 0.58, 0.44]

fig, ax = plt.subplots(figsize=(8, 5))
apply_base_style(fig, ax)

ax.plot(task_lengths, reliability, color=BASELINE, linewidth=2.5,
        marker="o", markersize=8, zorder=3, label="Frontier Model Reliability (SE)")
ax.fill_between(task_lengths, reliability, alpha=0.15, color=BASELINE)

for i, (label, val) in enumerate(zip(task_lengths, reliability)):
    ax.text(i, val + 0.015, f"{val:.2f}", ha="center", color=BASELINE,
            fontsize=11, fontweight="bold")

ax.axhline(y=0.44, color=BASELINE, linestyle=":", linewidth=1.2, alpha=0.6)
ax.set_ylabel("Reliability Score", fontsize=12)
ax.set_ylim(0.2, 1.05)
ax.set_title("Reliability Decay in LLM Agents as Task Horizon Grows\n"
             "Source: Khanal et al. 2026 — arXiv:2603.29231",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=GRID, labelcolor=TEXT, fontsize=11)

fig.tight_layout()
fig.savefig("charts/3_reliability_decay.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/3_reliability_decay.png")


# ── Chart 4: Hot-path latency overhead ────────────────────────────────────────
ops      = ["Entropy\nMonitor", "Shannon\nEntropy fn", "Loop\nDetector", "Loop\nTracker push"]
latencies = [0.0080, 0.0024, 0.0053, 0.0159]
sla       = 2.0

fig, ax = plt.subplots(figsize=(8, 5))
apply_base_style(fig, ax)

bars = ax.bar(ops, latencies, color=SOTIS, zorder=3, width=0.5)
ax.axhline(y=sla, color=BASELINE, linestyle="--", linewidth=1.8,
           label=f"SLA target ({sla} ms)", zorder=4)

for bar, val in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0003,
            f"{val:.4f} ms", ha="center", va="bottom",
            color=SOTIS, fontsize=10, fontweight="bold")

ax.set_ylabel("Average Latency (ms)", fontsize=12)
ax.set_ylim(0, 2.4)
ax.set_title("Sotis Hot-Path Overhead per Agent Step\n(N=10,000 calls · all ops ~100× under 2ms SLA)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(facecolor=GRID, labelcolor=TEXT, fontsize=11)

fig.tight_layout()
fig.savefig("charts/4_latency_overhead.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/4_latency_overhead.png")


# ── Chart 5: Token reduction — raw vs distilled ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
apply_base_style(fig, ax)

labels = ["Raw Trajectory\nTokens", "Sotis Resumption\nPrompt Tokens"]
values = [100, 12.27]
colors = [BASELINE, SOTIS]

bars = ax.bar(labels, values, color=colors, width=0.45, zorder=3)

# Labels above bars (in data coordinates only)
ax.text(bars[0].get_x() + bars[0].get_width()/2, 103,
        "100%", ha="center", va="bottom", color=BASELINE, fontsize=13, fontweight="bold")
ax.text(bars[1].get_x() + bars[1].get_width()/2, 15.5,
        "~12%", ha="center", va="bottom", color=SOTIS, fontsize=13, fontweight="bold")

# Arrow and label in data coordinates
ax.annotate("", xy=(0.78, 50), xytext=(0.22, 50),
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))
ax.text(0.5, 56, "87.7% reduction", ha="center", va="bottom",
        color=TEXT, fontsize=12, fontweight="bold")

ax.set_ylabel("Relative Token Count (%)", fontsize=12)
ax.set_ylim(0, 125)
ax.set_title("Context Distillation: Token Reduction\n(tiktoken BPE cl100k_base measurement)",
             fontsize=13, fontweight="bold", pad=14)

fig.tight_layout()
fig.savefig("charts/5_token_reduction.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/5_token_reduction.png")


# ── Chart 6: Real experiment results ──────────────────────────────────────────
# Data sourced directly from experimentLog/ run logs
experiments = [
    "AST Query Engine\n(Gemini 3.5)",
    "Circular Import\nTrap (Groq\nLlama 70B)",
    "Document\nHandling Loop\n(Mistral local)",
    "Circular Import\nTrap (OpenRouter\nGemini)",
]
steps        = [163, 78, 5, 60]   # approximate transitions from logs
resets       = [1,   1,  1,  1]
outcomes     = ["Rate-limited\n(meltdown caught)", "Completed", "Completed", "Completed"]
outcome_cols = [ACCENT, SOTIS, SOTIS, SOTIS]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1]})
apply_base_style(fig, ax_top)
ax_bot.set_visible(False)

x = np.arange(len(experiments))
bars = ax_top.bar(x, steps, color=outcome_cols, width=0.55, zorder=3)

for bar, r, o in zip(bars, resets, outcomes):
    ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{r} reset", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                o, ha="center", va="center", color=BG, fontsize=8.5, fontweight="bold",
                wrap=True)

ax_top.set_xticks(x)
ax_top.set_xticklabels(experiments, fontsize=9.5)
ax_top.set_ylabel("Agent Steps / Transitions", fontsize=12)
ax_top.set_ylim(0, 200)
ax_top.set_title("Real Agent Experiments — Meltdown Intercepts & Outcomes\n"
                 "(Gemini 3.5 · Groq Llama 70B · Mistral local · OpenRouter Gemini)",
                 fontsize=12, fontweight="bold", pad=14)

completed_patch = mpatches.Patch(color=SOTIS, label="Task Completed")
partial_patch   = mpatches.Patch(color=ACCENT, label="Meltdown Caught (API rate-limited)")
ax_top.legend(handles=[completed_patch, partial_patch],
              facecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper right")

fig.tight_layout()
fig.savefig("charts/6_real_experiments.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK charts/6_real_experiments.png")

print("\nAll charts saved to ./charts/ — ready for LinkedIn.")
