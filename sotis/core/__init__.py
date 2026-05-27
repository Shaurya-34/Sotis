"""
sotis.core — Pure computation layer.

Contains no LLM API dependencies. All modules in this package are
self-contained, deterministic, and fully unit-testable in isolation.
"""

from sotis.core.schemas import (
    Domain,
    ExecutionState,
    MeltdownReason,
    MeltdownSignal,
    SessionStatus,
    StepEvent,
    Subtask,
)
from sotis.core.entropy import (
    SessionEntropyTracker,
    EntropyMonitor,
    _shannon_entropy,
)
from sotis.core.loops import (
    LoopConfig,
    LoopResult,
    LoopDetector,
    LoopMonitor,
    SessionLoopTracker,
    WorkspaceDensityGuard,
)
from sotis.core.checkpoint import CheckpointManager, WorkspaceCheckpoint
from sotis.core.reset import ContextResetter
from sotis.core.decomposition import TaskDecomposer, verify_dag
from sotis.core.gds import calculate_gds, calculate_max_possible_gds

__all__ = [
    "Domain",
    "ExecutionState",
    "MeltdownReason",
    "MeltdownSignal",
    "SessionStatus",
    "StepEvent",
    "Subtask",
    "SessionEntropyTracker",
    "EntropyMonitor",
    "_shannon_entropy",
    "LoopConfig",
    "LoopResult",
    "LoopDetector",
    "LoopMonitor",
    "SessionLoopTracker",
    "WorkspaceDensityGuard",
    "CheckpointManager",
    "WorkspaceCheckpoint",
    "ContextResetter",
    "TaskDecomposer",
    "verify_dag",
    "calculate_gds",
    "calculate_max_possible_gds",
]
