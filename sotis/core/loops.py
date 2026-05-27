"""
sotis.core.loops
================
Exact tool-call loop detector for meltdown detection.

Algorithm Overview
------------------
A "loop" is defined as an identical (tool_name, args_hash) fingerprint
appearing REPEAT_THRESHOLD or more times within a WINDOW_SIZE step window.

Why this is necessary alongside entropy:
    Shannon entropy detects *variety* anomalies — the agent calling too many
    different tools randomly, or an agent stuck in a tight loop (e.g.,
    repeatedly calling `read_file` with identical arguments) might produce
    LOW entropy — it's always the same tool, so entropy alone would
    misclassify as "healthy".

    For loop detection, the complementary signal: it catches behavioural
    repetition regardless of overall entropy level.

Configuration: static per spec
    window_size       — Look-back N steps (default 6)
    repeat_threshold  — Flag if the same fingerprint appears ≥ N times (default 3)

Fingerprint = (tool_name, args_hash)
    This is intentionally truncated for efficiency; SHA-256
    collision probability over a step window is negligible.

Public API
----------
LoopConfig          : Frozen dataclass holding loop detection parameters.
LoopResult          : Output of `LoopDetector.evaluate()`, detailing detected loops.
LoopDetector        : Stateless, single-step evaluation.
SessionLoopTracker  : Stateful wrapper managing the sliding window for multi-step use.
WorkspaceDensityGuard : Tracks per-file consecutive edits without test outcome changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sotis.core.schemas import MeltdownReason, StepEvent


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoopConfig:
    """Configuration for the loop detector."""
    window_size: int = 6
    repeat_threshold: int = 3

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.repeat_threshold < 2:
            raise ValueError(f"repeat_threshold must be >= 2, got {self.repeat_threshold}")
        if self.repeat_threshold > self.window_size:
            raise ValueError(
                f"repeat_threshold cannot exceed window_size "
                f"({self.repeat_threshold} > {self.window_size})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    """
    Output of a loop evaluation pass.

    Attributes
    ----------
    meltdown_detected : True if any fingerprint exceeds repeat_threshold.
    looping_tools     : List of (tool_name, args_hash) pairs that exceeded threshold.
    dominant_loop     : The (tool_name, args_hash) with the highest count, or None.
    reason            : MeltdownReason.TOOL_LOOP if detected, else None.
    window_fingerprints : Fingerprints actually in the evaluation window.
    step_index        : Global step index of this evaluation (set by SessionLoopTracker).
    """
    meltdown_detected: bool = False
    looping_tools: List[Tuple[str, str]] = field(default_factory=list)
    dominant_loop: Optional[Tuple[str, str]] = None
    reason: Optional[MeltdownReason] = None
    window_fingerprints: List[Tuple[str, str]] = field(default_factory=list)
    step_index: int = 0

    @property
    def is_safe(self) -> bool:
        return not self.meltdown_detected

    @property
    def dominant_tool_name(self) -> Optional[str]:
        if self.dominant_loop is not None:
            return self.dominant_loop[0]
        return None

    @property
    def dominant_count(self) -> Optional[int]:
        """Return the occurrence count of the dominant loop fingerprint in the window."""
        if self.dominant_loop is not None:
            return Counter(self.window_fingerprints)[self.dominant_loop]
        return None


def _extract_query_text(event: StepEvent) -> Optional[str]:
    """Helper to extract search query or filter text from StepEvent arguments."""
    args = event.tool_args
    if not isinstance(args, dict):
        return None
    # Check standard search/query keys
    for key in ("query", "q", "filter_value", "search", "question", "filter"):
        val = args.get(key)
        if isinstance(val, str):
            return val
    return None


def _compute_jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token-based Jaccard similarity between two strings with robust normalization."""
    import re
    def normalize(s: str) -> str:
        s = s.replace("-", " ").replace("_", " ")
        s = re.sub(r"[^\w\s]", "", s)
        return s.lower()
        
    tokens1 = set(normalize(s1).split())
    tokens2 = set(normalize(s2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


# ─────────────────────────────────────────────────────────────────────────────
# LoopDetector (stateless)
# ─────────────────────────────────────────────────────────────────────────────

class LoopDetector:
    """
    Stateless loop detector. Evaluates a full trajectory and returns a LoopResult.
    """

    def __init__(self, config: Optional[LoopConfig] = None) -> None:
        self.config = config or LoopConfig()

    def evaluate(self, history: Sequence[StepEvent]) -> LoopResult:
        """
        Evaluate a sequence of StepEvents for tool loop patterns.

        Takes the last `window_size` events and checks if any unique (tool, args_hash)
        fingerprint appears `repeat_threshold` or more times, or if semantic similarity
        loops exist.
        """
        if not history:
            return LoopResult(meltdown_detected=False)

        # Look only at the sliding window
        window = list(history)[-self.config.window_size:]

        # Build fingerprints for the window
        fingerprints: List[Tuple[str, str]] = [
            (e.tool_name, e.args_hash) for e in window
        ]

        if len(window) < self.config.repeat_threshold:
            return LoopResult(meltdown_detected=False, window_fingerprints=fingerprints)

        # 1. Check for semantic query loops (Jaccard similarity >= 0.65)
        query_events = []
        for e in window:
            q_text = _extract_query_text(e)
            if q_text:
                query_events.append((e.tool_name, q_text, e))

        by_tool: Dict[str, List[str]] = {}
        for t_name, q_text, _ in query_events:
            by_tool.setdefault(t_name, []).append(q_text)

        for t_name, q_list in by_tool.items():
            if len(q_list) >= self.config.repeat_threshold:
                similar_count = 1
                for idx in range(len(q_list) - 1):
                    sim = _compute_jaccard_similarity(q_list[idx], q_list[idx + 1])
                    if sim >= 0.65:
                        similar_count += 1
                    else:
                        similar_count = 1
                if similar_count >= self.config.repeat_threshold:
                    dominant_event = next(e for t, q, e in query_events if t == t_name)
                    return LoopResult(
                        meltdown_detected=True,
                        looping_tools=[(t_name, dominant_event.args_hash)],
                        dominant_loop=(t_name, dominant_event.args_hash),
                        reason=MeltdownReason.TOOL_LOOP,
                        window_fingerprints=fingerprints,
                    )

        # 2. Traditional exact fingerprint loop detection
        counts = Counter(fingerprints)

        looping: List[Tuple[str, str]] = []
        for fp, count in counts.items():
            if count >= self.config.repeat_threshold:
                looping.append(fp)

        if not looping:
            return LoopResult(
                meltdown_detected=False,
                window_fingerprints=fingerprints,
            )

        # Identify dominant loop (highest count)
        dominant = max(looping, key=lambda fp: counts[fp])

        return LoopResult(
            meltdown_detected=True,
            looping_tools=looping,
            dominant_loop=dominant,
            reason=MeltdownReason.TOOL_LOOP,
            window_fingerprints=fingerprints,
        )

    def check_single_tool(self, tool_name: str, history: Sequence[StepEvent]) -> int:
        """Count how many times `tool_name` appears in the evaluation window."""
        window = list(history)[-self.config.window_size:]
        return sum(1 for e in window if e.tool_name == tool_name)


# ─────────────────────────────────────────────────────────────────────────────
# SessionLoopTracker (stateful)
# ─────────────────────────────────────────────────────────────────────────────

class SessionLoopTracker:
    """
    Stateful wrapper around LoopDetector.
    Maintains a rolling window of StepEvents and evaluates on each push.
    """

    def __init__(self, config: Optional[LoopConfig] = None) -> None:
        self.config = config or LoopConfig()
        self._detector = LoopDetector(self.config)
        self.window: List[StepEvent] = []
        self.latest_result: Optional[LoopResult] = None
        self._global_step: int = -1

    def push_event(self, event: StepEvent) -> LoopResult:
        """
        Add a new event, trim the window, and evaluate.
        Returns the LoopResult for the current state.
        """
        self._global_step += 1
        self.window.append(event)

        # Trim to window_size
        if len(self.window) > self.config.window_size:
            self.window = self.window[-self.config.window_size:]

        result = self._detector.evaluate(self.window)
        # Tag with global step index
        result.step_index = self._global_step
        self.latest_result = result
        return result

    def reset(self) -> None:
        """Clear all state for a fresh start (e.g., after a context reset)."""
        self.window.clear()
        self.latest_result = None
        self._global_step = -1

    @property
    def fingerprint_counts(self) -> Dict[Tuple[str, str], int]:
        """Return current fingerprint counts within the active window."""
        fingerprints = [(e.tool_name, e.args_hash) for e in self.window]
        return dict(Counter(fingerprints))


# ─────────────────────────────────────────────────────────────────────────────
# LoopMonitor (legacy alias — kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class LoopMonitor(LoopDetector):
    """Legacy alias for LoopDetector. Kept for backward compatibility."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# WorkspaceDensityGuard (Option B Extension)
# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceDensityGuard:
    """
    Tracks per-file consecutive edits without test outcome changes.

    Triggers a meltdown signal when an agent edits the same file
    `max_consecutive_edits` times without the test results changing
    (i.e., a "density loop" where the agent keeps editing but making no progress).

    Detection rules:
        - File modification tools: write_workspace_file, write_file, edit_file, save_file
        - Test execution tools: execute_workspace_tests, execute_tests, run_tests, pytest
        - If a test is run and the result_summary differs from the previous run,
          all edit counters are reset (the agent made progress).
        - If a test is run and the result_summary is identical, counters are preserved.
    """

    EDIT_TOOLS = {"write_workspace_file", "write_file", "edit_file", "save_file"}
    TEST_TOOLS = {"execute_workspace_tests", "execute_tests", "run_tests", "pytest"}

    def __init__(self, max_consecutive_edits: int = 3) -> None:
        self.max_consecutive_edits = max_consecutive_edits
        self._edit_counts: Dict[str, int] = {}     # file_path -> consecutive edit count
        self._last_test_result: Optional[str] = None
        self.latest_meltdown: bool = False
        self.triggered_file: Optional[str] = None

    def push_event(self, event: StepEvent) -> bool:
        """
        Process a new event. Returns True if meltdown threshold is reached.
        """
        tool = event.tool_name

        # Check if this is a file edit tool
        if tool in self.EDIT_TOOLS:
            # Extract file path from tool_args
            file_path = (
                event.tool_args.get("file_path")
                or event.tool_args.get("path")
                or event.tool_args.get("filename")
            )
            if file_path:
                self._edit_counts[file_path] = self._edit_counts.get(file_path, 0) + 1

                if self._edit_counts[file_path] >= self.max_consecutive_edits:
                    self.latest_meltdown = True
                    self.triggered_file = file_path
                    return True

        # Check if this is a test execution tool
        elif tool in self.TEST_TOOLS:
            current_result = event.result_summary
            if (
                current_result is not None
                and self._last_test_result is not None
                and current_result != self._last_test_result
            ):
                # Test outcome changed — agent is making progress, reset all counters
                self._edit_counts.clear()
            self._last_test_result = current_result

        return False

    def reset(self) -> None:
        """Reset the density guard's state (e.g., after a rollback or meltdown recovery)."""
        self._edit_counts.clear()
        self._last_test_result = None
        self.latest_meltdown = False
        self.triggered_file = None

