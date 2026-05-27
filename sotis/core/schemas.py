"""
sotis.core.schemas
==================
Canonical data structures shared across the entire Sotis engine.

Design philosophy:
    - Immutable by default (frozen Pydantic models) to prevent accidental mutation.
    - All fields are type-annotated and validated on construction.
    - Hashing of tool arguments is handled here so downstream modules
      (entropy, loops, checkpoint) never deal with raw dict comparison.

Core types
----------
StepEvent       : A single agent tool invocation captured by the monitor.
MeltdownSignal  : The structured output emitted when a meltdown is detected.
Subtask         : One node in the task decomposition graph (Phase 3).
ExecutionState  : The mutable state object threaded through an entire session.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    """Task domain used to select domain-appropriate decomposition strategies."""
    SOFTWARE_ENGINEERING = "SE"
    WEB_RESEARCH         = "WR"
    DOCUMENT_PROCESSING  = "DP"
    UNKNOWN              = "UNKNOWN"


class MeltdownReason(str, Enum):
    """Categorical reason the meltdown detector fired."""
    HIGH_ENTROPY       = "HIGH_ENTROPY"        # H(t) exceeded static threshold
    ENTROPY_TREND      = "ENTROPY_TREND"       # H(t) rising for 3+ consecutive steps
    TOOL_LOOP          = "TOOL_LOOP"           # Exact (tool, args_hash) pair repeated ≥3×
    COMBINED           = "COMBINED"            # Both entropy and loop signals active
    BUDGET_EXCEEDED    = "BUDGET_EXCEEDED"     # Step count exhausted for subtask


class SessionStatus(str, Enum):
    """High-level status of the overall execution session."""
    RUNNING       = "RUNNING"
    MELTDOWN      = "MELTDOWN"
    RESETTING     = "RESETTING"
    RESUMED       = "RESUMED"
    HARD_FAILED   = "HARD_FAILED"   # Max resets (2) exhausted for a subtask
    COMPLETED     = "COMPLETED"


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class StepEvent(BaseModel):
    """
    A single, atomic tool invocation captured by the Sotis monitor.

    Every call an agent makes — read_file, search_web, write_code, etc. —
    is recorded as a StepEvent and fed through the entropy and loop detectors.

    Fields
    ------
    step_index      : Monotonically increasing integer across the session.
    timestamp_ms    : Unix epoch milliseconds of the event.
    tool_name       : The name of the tool invoked (e.g. "read_file").
    tool_args       : Raw arguments dict passed to the tool.
    args_hash       : SHA-256 hex digest of the JSON-serialized tool_args.
                      Computed automatically — used by the loop detector for O(1) lookups.
    result_summary  : Optional short summary of the tool result (e.g. first 200 chars).
    subtask_id      : Identifier of the active subtask at the time of this event.
    """
    model_config = {"frozen": True}   # Immutable after construction.

    step_index    : int               = Field(..., ge=0, description="Session-global step counter.")
    timestamp_ms  : float             = Field(default_factory=lambda: time.time() * 1000)
    tool_name     : str               = Field(..., min_length=1)
    tool_args     : Dict[str, Any]    = Field(default_factory=dict)
    args_hash     : str               = Field(default="")
    result_summary: Optional[str]     = Field(default=None, max_length=500)
    subtask_id    : Optional[str]     = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def compute_args_hash(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the SHA-256 hash of ``tool_args`` when not explicitly provided.
        Hashing is key-sorted to make identical argument dicts always produce
        the same hash regardless of insertion order.
        """
        if not values.get("args_hash"):
            args = values.get("tool_args", {})
            serialized = json.dumps(args, sort_keys=True, ensure_ascii=False)
            values["args_hash"] = hashlib.sha256(serialized.encode()).hexdigest()
        return values

    @property
    def tool_fingerprint(self) -> str:
        """
        A compact (tool_name, args_hash) identifier used by the loop detector.
        Format: ``<tool_name>:<first-16-chars-of-args-hash>``
        """
        return f"{self.tool_name}:{self.args_hash[:16]}"


class MeltdownSignal(BaseModel):
    """
    Emitted by the entropy monitor or loop detector when a meltdown is detected.

    Consumed by:
        - CheckpointManager  (Phase 2) to freeze and snapshot state
        - ContextResetter    (Phase 2) to build the distilled resumption prompt
        - Logger             (Phase 4) to record the event in the session ledger
    """
    model_config = {"frozen": True}

    session_id        : str
    subtask_id        : Optional[str]
    triggered_at_step : int
    reason            : MeltdownReason
    entropy_value     : Optional[float]    = None   # H(t) at time of trigger
    loop_tool         : Optional[str]      = None   # Tool name in loop (if TOOL_LOOP)
    loop_count        : Optional[int]      = None   # How many repeats were detected
    reset_attempt     : int                = 1      # Which reset attempt this is (1 or 2)
    timestamp_ms      : float              = Field(default_factory=lambda: time.time() * 1000)

    @field_validator("reset_attempt")
    @classmethod
    def cap_reset_attempts(cls, v: int) -> int:
        if v < 1 or v > 2:
            raise ValueError(
                f"reset_attempt must be 1 or 2 (hard cap). Got {v}. "
                "If both resets have been exhausted, the subtask should be "
                "marked HARD_FAILED at the session level."
            )
        return v


class Subtask(BaseModel):
    """
    One node in the upstream task decomposition graph (Phase 3).
    Included in schemas now because StepEvent references subtask_id.

    Fields
    ------
    subtask_id      : Unique slug for this subtask within the session.
    description     : Human-readable goal of this subtask.
    domain          : Task domain (SE, WR, DP).
    step_budget     : Maximum number of steps allocated to this subtask.
    gds_weight      : Fractional weight used for GDS calculation (all weights must sum to 1.0).
    dependencies    : Ordered list of subtask_ids that must complete before this one starts.
    status          : Current lifecycle status.
    completed_steps : Number of steps consumed so far.
    resets_used     : Number of context resets consumed for this subtask (max 2).
    """
    subtask_id     : str
    description    : str                   = Field(..., min_length=1)
    domain         : Domain                = Domain.UNKNOWN
    step_budget    : int                   = Field(default=60, ge=1, le=200)
    gds_weight     : float                 = Field(default=0.25, ge=0.0, le=1.0)
    dependencies   : List[str]             = Field(default_factory=list)
    status         : str                   = "PENDING"  # PENDING, ACTIVE, DONE, FAILED
    completed_steps: int                   = 0
    resets_used    : int                   = Field(default=0, ge=0, le=2)

    def is_step_budget_exhausted(self) -> bool:
        return self.completed_steps >= self.step_budget

    def can_reset(self) -> bool:
        """Returns True if at least one more context reset is available."""
        return self.resets_used < 2


class ExecutionState(BaseModel):
    """
    Mutable session-level state threaded through the entire Sotis runtime.

    This is the single source of truth for the monitor. The entropy calculator,
    loop detector, checkpoint manager, and context resetter all read from and
    write to this object.

    NOT frozen — designed to be mutated in place as events arrive.
    """
    session_id    : str
    domain        : Domain                 = Domain.UNKNOWN
    status        : SessionStatus          = SessionStatus.RUNNING
    subtasks      : List[Subtask]          = Field(default_factory=list)
    active_subtask_id: Optional[str]       = None
    trajectory    : List[StepEvent]        = Field(default_factory=list)
    meltdown_log  : List[MeltdownSignal]   = Field(default_factory=list)
    created_at_ms : float                  = Field(default_factory=lambda: time.time() * 1000)
    total_resets  : int                    = 0

    @property
    def step_count(self) -> int:
        return len(self.trajectory)

    @property
    def active_subtask(self) -> Optional[Subtask]:
        if self.active_subtask_id is None:
            return None
        return next(
            (s for s in self.subtasks if s.subtask_id == self.active_subtask_id),
            None,
        )

    def record_step(self, event: StepEvent) -> None:
        """Append a new StepEvent to the session trajectory."""
        self.trajectory.append(event)

    def record_meltdown(self, signal: MeltdownSignal) -> None:
        """Log a detected meltdown signal."""
        self.meltdown_log.append(signal)
        self.total_resets += 1
        self.status = SessionStatus.MELTDOWN

    def get_window(self, window_size: int) -> List[StepEvent]:
        """Return the most recent ``window_size`` step events."""
        return self.trajectory[-window_size:]
