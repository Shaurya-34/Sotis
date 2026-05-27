"""
sotis.obs.logger
================
Structured telemetry logger recording events to a JSON-line session file.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from sotis.core.schemas import StepEvent, MeltdownSignal, ExecutionState


class SessionLogger:
    """
    Orchestrates real-time telemetry logging of StepEvents, MeltdownSignals,
    and ExecutionState snapshots to a single session JSON-line file.
    """

    def __init__(self, session_id: str, log_dir: str = "logs") -> None:
        self.session_id = session_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"session_{session_id}.json")

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Appends a single JSON event to the session log file."""
        entry = {
            "session_id": self.session_id,
            "timestamp_ms": time.time() * 1000,
            "event_type": event_type,
            "data": data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_step(self, step: StepEvent) -> None:
        """Records a single StepEvent tool invocation."""
        self.log_event("step", step.model_dump())

    def log_meltdown(self, signal: MeltdownSignal) -> None:
        """Records a detected meltdown signal."""
        self.log_event("meltdown", signal.model_dump())

    def log_state(self, state: ExecutionState) -> None:
        """Records a summary snapshot of overall execution state."""
        self.log_event(
            "state",
            {
                "status": state.status.value,
                "active_subtask_id": state.active_subtask_id,
                "total_resets": state.total_resets,
                "step_count": len(state.trajectory),
                "subtasks": [s.model_dump() for s in state.subtasks],
            },
        )
