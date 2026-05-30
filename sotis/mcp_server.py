"""
sotis.mcp_server
================
Model Context Protocol server for Sotis.

Exposes Sotis's meltdown detection to any MCP-capable agent (Claude Code,
Claude Desktop, etc.). The agent reports each tool call to ``sotis_watch`` and
Sotis replies with the live entropy reading and a meltdown verdict — while
streaming the same telemetry to ``logs/session_sotis-mcp-*.json`` so the
dashboard can render it live.

Run with:  ``sotis mcp``  (stdio transport)

Wire it into Claude Code via ``.mcp.json``::

    {
      "mcpServers": {
        "sotis": { "command": "sotis", "args": ["mcp"] }
      }
    }

Then tell the agent (e.g. in CLAUDE.md) to call ``sotis_watch`` after every
tool use.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from sotis import SotisGuard
from sotis.core.schemas import StepEvent
from sotis.obs.logger import SessionLogger

mcp = FastMCP("sotis")


# ─── Session state ────────────────────────────────────────────────────────────

class _McpSession:
    """
    A single live monitoring session. Drives the low-level trackers directly so
    it can surface the real entropy value and the specific meltdown reason
    (TOOL_LOOP / ENTROPY_PEAK / EDIT_DENSITY).
    """

    def __init__(self, task_goal: str) -> None:
        self.session_id   = f"sotis-mcp-{uuid.uuid4().hex[:8]}"
        self.task_goal    = task_goal
        self.guard        = SotisGuard()
        self.logger       = SessionLogger(self.session_id)
        self.step_index   = 0
        self.total_resets = 0
        self.status       = "RUNNING"
        self.last_entropy = 0.0

    def watch(self, tool_name: str, tool_args: Dict[str, Any],
              result_summary: Optional[str]) -> Dict[str, Any]:
        self.step_index += 1
        event = StepEvent(
            step_index=self.step_index,
            tool_name=tool_name,
            tool_args=tool_args,
            result_summary=result_summary,
        )

        # Drive each detector directly to capture rich detail.
        entropy_res = self.guard.entropy_tracker.push_event(event)
        loop_res    = self.guard.loop_tracker.push_event(event)
        density_res = self.guard.density_guard.push_event(event)
        if entropy_res.entropy is not None:
            self.last_entropy = entropy_res.entropy

        self.logger.log_event("step", {
            "step_index":     self.step_index,
            "tool_name":      tool_name,
            "tool_args":      tool_args,
            "result_summary": result_summary,
        })

        # Reason priority: exact loop > edit density > high entropy.
        meltdown = False
        reason: Optional[str] = None
        if loop_res.meltdown_detected:
            meltdown, reason = True, "TOOL_LOOP"
        elif density_res:
            meltdown, reason = True, "EDIT_DENSITY"
        elif entropy_res.meltdown_detected:
            meltdown, reason = True, "ENTROPY_PEAK"

        if meltdown:
            self.total_resets += 1
            self.logger.log_event("meltdown", {
                "triggered_at_step": self.step_index,
                "reason":            reason,
                "entropy_value":     entropy_res.entropy,
                "loop_tool":         loop_res.dominant_tool_name,
            })
            self.guard.reset()
            self.status = "RESUMED"

        self._emit_state()
        return {
            "meltdown":      meltdown,
            "reason":        reason,
            "entropy":       self.last_entropy,
            "step":          self.step_index,
            "total_resets":  self.total_resets,
            "status":        self.status,
        }

    def manual_reset(self) -> None:
        self.guard.reset()
        self.total_resets += 1
        self.status = "RESUMED"
        self._emit_state()

    def _emit_state(self) -> None:
        self.logger.log_event("state", {
            "status":       self.status,
            "total_resets": self.total_resets,
            "step_count":   self.step_index,
            "subtasks":     [],
        })

    def close(self) -> None:
        self.status = "COMPLETED"
        self._emit_state()


# Module-level singleton — Claude Code is a single agent, so one session at a time.
_active: Optional[_McpSession] = None


def _parse_args(raw: str) -> Dict[str, Any]:
    """tool_args arrives as a JSON string from the agent; tolerate plain text."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def sotis_start_session(task_goal: str) -> str:
    """
    Begin a new Sotis monitoring session for the current task.

    Call this once at the start of a task. It opens a telemetry stream that the
    Sotis dashboard (`sotis dashboard`, Live Mode on) renders in real time.

    Args:
        task_goal: A short description of what the agent is trying to accomplish.

    Returns:
        The session id and confirmation.
    """
    global _active
    if _active is not None:
        _active.close()
    _active = _McpSession(task_goal)
    return (
        f"Sotis session started: {_active.session_id}\n"
        f"Goal: {task_goal}\n"
        f"Open the dashboard with `sotis dashboard` and enable Live Mode to watch.\n"
        f"Report each tool call to `sotis_watch` so Sotis can detect meltdowns."
    )


@mcp.tool()
def sotis_watch(tool_name: str, tool_args: str = "", result_summary: str = "") -> str:
    """
    Report a tool call to Sotis and get a live meltdown verdict.

    Call this immediately AFTER every tool you use. Sotis tracks the diversity of
    your recent actions; if you start looping or thrashing, it flags a meltdown
    and you should reconsider your approach (re-read the task, change strategy).

    Args:
        tool_name: The name of the tool that was just called (e.g. "read_file").
        tool_args: The arguments, as a JSON string. Optional.
        result_summary: A one-line summary of the result. Optional.

    Returns:
        A status line with the entropy reading and meltdown verdict.
    """
    global _active
    if _active is None:
        _active = _McpSession("(auto-started)")

    res = _active.watch(tool_name, _parse_args(tool_args), result_summary or None)

    if res["meltdown"]:
        return (
            f"MELTDOWN ({res['reason']}) at step {res['step']} "
            f"— entropy {res['entropy']:.3f} bits.\n"
            f"Sotis has reset the monitor (reset #{res['total_resets']}). "
            f"Stop repeating the same action: re-read the task goal and try a "
            f"different approach before continuing."
        )
    return (
        f"OK step {res['step']} — entropy {res['entropy']:.3f} bits, "
        f"status {res['status']}, resets {res['total_resets']}."
    )


@mcp.tool()
def sotis_status() -> str:
    """
    Get the current Sotis session metrics (status, step count, resets, entropy).

    Returns:
        A human-readable status summary, or a note if no session is active.
    """
    if _active is None:
        return "No active Sotis session. Call sotis_start_session first."
    return (
        f"Session {_active.session_id}\n"
        f"Goal:    {_active.task_goal}\n"
        f"Status:  {_active.status}\n"
        f"Steps:   {_active.step_index}\n"
        f"Resets:  {_active.total_resets}\n"
        f"Entropy: {_active.last_entropy:.3f} bits"
    )


@mcp.tool()
def sotis_reset() -> str:
    """
    Manually reset Sotis's monitors — clears the rolling tool-history window.

    Use this when you have deliberately changed strategy and want the entropy
    detector to start fresh rather than penalising your earlier exploration.

    Returns:
        Confirmation of the reset.
    """
    if _active is None:
        return "No active Sotis session. Call sotis_start_session first."
    _active.manual_reset()
    return f"Sotis monitors reset (reset #{_active.total_resets}). Window cleared."


def main() -> None:
    """Entry point for `sotis mcp` — runs the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
