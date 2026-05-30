"""
Sotis — MOP-Triggered Context Reset Agent
==========================================
Reliability middleware that monitors long-horizon LLM agent sessions,
detects meltdowns via sliding-window Shannon entropy analysis, and
transparently resets context to restore forward progress.

Based on: "Beyond pass@1: A Reliability Science Framework for
           Long-Horizon LLM Agents" (arXiv:2603.29231v1)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from sotis.core.schemas import StepEvent, MeltdownSignal, Domain, MeltdownReason
from sotis.core.entropy import SessionEntropyTracker, EntropyConfig
from sotis.core.loops import SessionLoopTracker, LoopConfig, WorkspaceDensityGuard

__version__ = "1.0.2"
__author__ = "Sotis Contributors"


class SotisGuard:
    """
    SotisGuard
    ==========
    High-level user-facing developer facade designed to watch agent trajectories
    and detect meltdown conditions in 3 lines of code.

    Usage:
    ------
    >>> from sotis import SotisGuard
    >>> guard = SotisGuard()
    >>> meltdown = guard.watch("write_file", {"file_path": "app.py"}, "Written successfully")
    """

    def __init__(
        self,
        entropy_config: Optional[EntropyConfig] = None,
        loop_config: Optional[LoopConfig] = None,
        max_consecutive_edits: int = 3,
    ) -> None:
        self.entropy_tracker = SessionEntropyTracker(entropy_config)
        self.loop_tracker = SessionLoopTracker(loop_config)
        self.density_guard = WorkspaceDensityGuard(max_consecutive_edits)
        self._step_counter = 0

    def watch(
        self,
        tool_name: Union[str, StepEvent],
        tool_args: Optional[Dict[str, Any]] = None,
        result_summary: Optional[str] = None,
        subtask_id: Optional[str] = None,
    ) -> bool:
        """
        Record a step event and evaluate it for meltdown patterns.

        Returns True if a tool-loop, high-entropy, or edit-density meltdown is triggered.
        """
        if isinstance(tool_name, StepEvent):
            event = tool_name
        else:
            event = StepEvent(
                step_index=self._step_counter,
                tool_name=tool_name,
                tool_args=tool_args or {},
                result_summary=result_summary,
                subtask_id=subtask_id,
            )
            self._step_counter += 1

        # Push to all three monitors
        entropy_res = self.entropy_tracker.push_event(event)
        loop_res = self.loop_tracker.push_event(event)
        density_res = self.density_guard.push_event(event)

        # Returns True if any meltdown is triggered
        return (
            entropy_res.meltdown_detected or
            loop_res.meltdown_detected or
            density_res
        )

    def reset(self) -> None:
        """Resets the state of all monitors."""
        self.entropy_tracker.reset()
        self.loop_tracker.reset()
        self.density_guard.reset()
        self._step_counter = 0


__all__ = [
    "SotisGuard",
    "StepEvent",
    "MeltdownSignal",
    "Domain",
    "MeltdownReason",
    "EntropyConfig",
    "LoopConfig",
]
