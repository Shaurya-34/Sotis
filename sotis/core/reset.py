"""
sotis.core.reset
================
Context distiller and resumption prompt builder.

The core problem
----------------
When an agent meltdown occurs, the conversation trajectory can be tens of
thousands of tokens long. Naively injecting the entire history into the new
context window wastes tokens, re-exposes the agent to the confusing spiral
it just escaped, and provides no clarity about what was already achieved.

Sotis solves this with "context distillation": instead of replaying history,
a compact, structured resumption prompt is synthesised from:
    1. The original task goal.
    2. A list of fully-completed subtasks (with their GDS-weighted progress).
    3. The current subtask's goal and step budget.
    4. A summary of the workspace state from the checkpoint (modified files).
    5. The last N unique tool-call observations from the trajectory.
    6. Explicit guidance to continue from the current state.

Token reduction target: >= 60% fewer tokens than the raw trajectory.

Measurement
-----------
Token counts are approximated using a simple character / 4 ratio (common
GPT-3.5/GPT-4 approximation). This avoids adding a tiktoken dependency while
remaining accurate enough for the percentage ratio metric.

Public API
----------
DistillationConfig  : Configuration for the distiller.
DistillationResult  : Output of ContextResetter.distill() — contains the
                      resumption prompt text AND the token reduction ratio.
ContextResetter     : Stateless distiller. Call distill() per reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sotis.core.checkpoint import WorkspaceCheckpoint
from sotis.core.schemas import ExecutionState, MeltdownReason


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DistillationConfig:
    """
    Configuration for the context distiller.

    Attributes
    ----------
    max_observations      : Maximum number of unique tool observations to
                            include in the resumption prompt. Default: 8.
    include_file_diffs    : If True, include a summary of modified files from
                            the checkpoint. Default: True.
    include_warning_note  : If True, prepend a brief note about why the reset
                            occurred (entropy spike / loop). Default: True.
    max_diff_lines_per_file: Truncate large diffs to this many lines to keep
                             the prompt compact. Default: 30.
    """
    max_observations        : int  = 8
    include_file_diffs      : bool = True
    include_warning_note    : bool = True
    max_diff_lines_per_file : int  = 30


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillationResult:
    """
    Output of ContextResetter.distill().

    Attributes
    ----------
    prompt              : The complete distilled resumption system prompt.
    raw_token_estimate  : Estimated token count of the original trajectory.
    distilled_token_estimate: Estimated token count of the distilled prompt.
    token_reduction_pct : Percentage reduction in token count.
                          Formula: (1 - distilled/raw) * 100
    reset_number        : Which reset attempt this is (1 or 2).
    session_id          : Source session identifier.
    """
    prompt                   : str
    raw_token_estimate       : int
    distilled_token_estimate : int
    token_reduction_pct      : float
    reset_number             : int
    session_id               : str

    @property
    def meets_reduction_target(self) -> bool:
        """Returns True if the distilled prompt is ≥ 60% smaller than raw."""
        return self.token_reduction_pct >= 60.0

    def summary_line(self) -> str:
        return (
            f"[Reset #{self.reset_number}] "
            f"Token reduction: {self.token_reduction_pct:.1f}% "
            f"({self.raw_token_estimate} → {self.distilled_token_estimate} est. tokens). "
            f"Target met: {self.meets_reduction_target}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation helper
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using OpenAI's cl100k_base tiktoken BPE encoder."""
        return len(_enc.encode(text))
except ImportError:
    def _estimate_tokens(text: str) -> int:
        """Approximate token count using the chars/4 heuristic as a fallback."""
        return max(1, len(text) // 4)


def _truncate_diff(diff_text: str, max_lines: int) -> str:
    """Truncate a diff to at most max_lines lines, appending a note if cut."""
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    truncated = lines[:max_lines]
    truncated.append(f"... [{len(lines) - max_lines} more lines truncated] ...")
    return "\n".join(truncated)


# ─────────────────────────────────────────────────────────────────────────────
# Context Resetter
# ─────────────────────────────────────────────────────────────────────────────

class ContextResetter:
    """
    Stateless context distiller.

    Takes the current ``ExecutionState`` and a ``WorkspaceCheckpoint`` and
    produces a compact resumption system prompt that gives the agent everything
    it needs to continue work without re-reading the raw trajectory.

    Stateless by design — the runtime calls distill() each time a reset fires.
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        self.config = config or DistillationConfig()

    def distill(
        self,
        state       : ExecutionState,
        checkpoint  : WorkspaceCheckpoint,
        task_goal   : str,
    ) -> DistillationResult:
        """
        Produce a distilled resumption prompt from the current session state.

        Parameters
        ----------
        state       : Current ExecutionState (trajectory, subtasks).
        checkpoint  : WorkspaceCheckpoint produced by CheckpointManager.
        task_goal   : The original high-level task description.

        Returns
        -------
        DistillationResult containing the prompt and token metrics.
        """
        cfg = self.config

        # 1. Estimate raw trajectory token cost.
        raw_trajectory_text = self._serialise_raw_trajectory(state)
        raw_tokens          = _estimate_tokens(raw_trajectory_text)

        # 2. Build each section of the distilled prompt.
        sections: List[str] = []

        if cfg.include_warning_note:
            sections.append(self._build_warning_section(checkpoint))

        sections.append(self._build_goal_section(task_goal))
        sections.append(self._build_progress_section(state, checkpoint))
        sections.append(self._build_current_subtask_section(state))
        sections.append(self._build_observations_section(state, cfg.max_observations))

        if cfg.include_file_diffs and checkpoint.file_diffs:
            sections.append(self._build_workspace_section(checkpoint, cfg.max_diff_lines_per_file))

        sections.append(self._build_continuation_section())

        prompt           = "\n\n".join(s for s in sections if s.strip())
        distilled_tokens = _estimate_tokens(prompt)

        reduction_pct = (
            (1.0 - distilled_tokens / raw_tokens) * 100.0
            if raw_tokens > 0 else 0.0
        )

        return DistillationResult(
            prompt=prompt,
            raw_token_estimate=raw_tokens,
            distilled_token_estimate=distilled_tokens,
            token_reduction_pct=round(reduction_pct, 2),
            reset_number=checkpoint.reset_number,
            session_id=state.session_id,
        )

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_warning_section(self, checkpoint: WorkspaceCheckpoint) -> str:
        reason_text = {
            MeltdownReason.HIGH_ENTROPY.value : "high tool-call entropy (disorganised execution pattern)",
            MeltdownReason.ENTROPY_TREND.value: "a sustained rising entropy trend (early meltdown signal)",
            MeltdownReason.TOOL_LOOP.value    : "repeated identical tool calls (execution loop detected)",
            MeltdownReason.COMBINED.value     : "both elevated entropy and repeated tool calls",
            MeltdownReason.BUDGET_EXCEEDED.value: "the step budget for the current subtask being exhausted",
        }.get(checkpoint.meltdown_reason, checkpoint.meltdown_reason)

        return (
            f"## Context Reset Notice (Reset #{checkpoint.reset_number}/2)\n"
            f"The previous execution context was reset after detecting {reason_text} "
            f"at step {checkpoint.snapshot_at_step}. "
            f"Your verified progress has been preserved. "
            f"Continue from where you left off — do NOT restart the task from scratch."
        )

    def _build_goal_section(self, task_goal: str) -> str:
        return f"## Task Goal\n{task_goal.strip()}"

    def _build_progress_section(
        self,
        state     : ExecutionState,
        checkpoint: WorkspaceCheckpoint,
    ) -> str:
        lines = ["## Verified Progress"]

        if checkpoint.completed_subtasks:
            lines.append("### Completed Subtasks")
            for st_id in checkpoint.completed_subtasks:
                # Look up the subtask object for its description.
                st_obj = next((s for s in state.subtasks if s.subtask_id == st_id), None)
                desc   = st_obj.description if st_obj else st_id
                weight = f"{st_obj.gds_weight * 100:.0f}%" if st_obj else "?"
                lines.append(f"  [DONE] {desc}  (GDS weight: {weight})")
        else:
            lines.append("  No subtasks fully completed yet.")

        total_steps = state.step_count
        lines.append(f"\nTotal steps executed so far: {total_steps}")
        lines.append(f"Total context resets used: {state.total_resets}/2")
        return "\n".join(lines)

    def _build_current_subtask_section(self, state: ExecutionState) -> str:
        active = state.active_subtask
        if not active:
            return "## Current Subtask\nNo active subtask — determine the next goal from the task structure."

        lines = [
            "## Current Subtask (Resume Here)",
            f"Goal:         {active.description}",
            f"Domain:       {active.domain.value}",
            f"Step budget:  {active.step_budget} steps "
            f"({active.completed_steps} used, "
            f"{max(0, active.step_budget - active.completed_steps)} remaining)",
            f"Resets used:  {active.resets_used}/2",
        ]
        if active.dependencies:
            lines.append(f"Depends on:   {', '.join(active.dependencies)}")
        return "\n".join(lines)

    def _build_observations_section(
        self,
        state        : ExecutionState,
        max_obs      : int,
    ) -> str:
        """Extract the last N unique, informative tool observations."""
        lines = ["## Key Observations from Previous Execution"]

        # Walk backward through trajectory, collecting unique result summaries.
        seen_tools: dict = {}
        for event in reversed(state.trajectory):
            if event.result_summary and event.tool_name not in seen_tools:
                seen_tools[event.tool_name] = event.result_summary
                if len(seen_tools) >= max_obs:
                    break

        if not seen_tools:
            lines.append("  No tool results recorded yet.")
        else:
            for tool, result in seen_tools.items():
                # Truncate long results.
                snippet = result[:200].replace("\n", " ")
                lines.append(f"  [{tool}] → {snippet}")

        return "\n".join(lines)

    def _build_workspace_section(
        self,
        checkpoint    : WorkspaceCheckpoint,
        max_diff_lines: int,
    ) -> str:
        lines = ["## Workspace State at Reset Point"]

        if checkpoint.modified_files:
            lines.append(f"Modified files ({len(checkpoint.modified_files)}):")
            for path in checkpoint.modified_files:
                diff = checkpoint.file_diffs[path]
                lines.append(f"  ~ {path}  (+{diff.lines_added} / -{diff.lines_removed})")
                if diff.unified_diff:
                    truncated = _truncate_diff(diff.unified_diff, max_diff_lines)
                    # Indent the diff for readability.
                    indented  = "\n".join("    " + l for l in truncated.splitlines())
                    lines.append(indented)

        if checkpoint.added_files:
            lines.append(f"\nNew files created ({len(checkpoint.added_files)}):")
            for path in checkpoint.added_files:
                lines.append(f"  + {path}")

        if checkpoint.deleted_files:
            lines.append(f"\nDeleted files ({len(checkpoint.deleted_files)}):")
            for path in checkpoint.deleted_files:
                lines.append(f"  - {path}")

        if not checkpoint.file_diffs:
            lines.append("  No file changes detected in tracked paths.")

        return "\n".join(lines)

    def _build_continuation_section(self) -> str:
        return (
            "## Instructions\n"
            "1. Resume work on the **Current Subtask** listed above.\n"
            "2. Do NOT repeat steps already completed — build on the verified progress.\n"
            "3. If you previously encountered an obstacle, approach it differently.\n"
            "4. Stay focused — use only the tools necessary for the current subtask.\n"
            "5. Complete the subtask within the remaining step budget.\n"
            "6. CRITICAL: Always format your tool/function calls strictly using the standard JSON tool-calling blocks. "
            "Do NOT output raw XML-like tags (e.g. <function=...>) under any circumstances."
        )

    # ── Raw trajectory serialiser (for token estimation only) ─────────────────

    def _serialise_raw_trajectory(self, state: ExecutionState) -> str:
        """
        Produce a compact text representation of the full trajectory.
        Used only to estimate the 'before' token count for the reduction ratio.
        """
        parts = []
        for e in state.trajectory:
            part = f"Step {e.step_index}: {e.tool_name}({e.args_hash[:8]})"
            if e.result_summary:
                part += f" -> {e.result_summary[:100]}"
            parts.append(part)
        return "\n".join(parts)
