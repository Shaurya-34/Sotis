"""
sotis.core.entropy
==================
Sliding-window Shannon entropy monitor for real-time meltdown detection.

Algorithm Overview
------------------
At every step, Sotis computes the Shannon entropy H(t) of the distribution
of tool calls over the last N steps (default N=5). High entropy indicates
the agent is calling many different tools in a disorganised, unpredictable
pattern. Low entropy indicates healthy, focused tool usage.

Shannon entropy is defined as:
    H(t) = -Σ p(x) * log₂(p(x))   for each unique tool x in the window

Static thresholds (configured at construction time):
    window_size   : N = 5  (number of steps in the sliding window)
    hard_threshold: H_max = 1.5   → immediate meltdown trigger
    trend_window  : 3 consecutive steps where H is increasing → early warning

Design decisions
----------------
- Pure Python + NumPy only. No LLM API calls.
- Completely stateless: the monitor takes a list of StepEvent objects as
  input and returns a result. No internal mutation.
- The "trend detector" adds a secondary early warning before the hard
  threshold is breached, matching the paper's observation that meltdowns
  onset gradually before becoming visible.

Public API
----------
EntropyConfig       : Dataclass holding static threshold configuration.
EntropyResult       : Returned by EntropyMonitor.evaluate(); carries H(t),
                      warning state, and meltdown flag.
EntropyMonitor      : The primary callable. Stateless; call evaluate() per step.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sotis.core.schemas import MeltdownReason, StepEvent


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntropyConfig:
    """
    Static configuration for the sliding-window entropy monitor.

    Attributes
    ----------
    window_size       : Number of most-recent steps included in H(t) calculation.
                        Approved: N = 5.
    hard_threshold    : H(t) value above which a meltdown is immediately triggered.
                        Approved: H_max = 1.5.
    trend_steps       : Number of consecutive steps with strictly increasing H(t)
                        before an ENTROPY_TREND early warning is raised.
                        Default: 3.
    min_window_fill   : Minimum number of steps that must exist before evaluation
                        begins. Below this, no signal is emitted.
                        Default: 3 (avoids false positives at session start).
    """
    window_size    : int   = 5
    hard_threshold : float = 1.5
    trend_steps    : int   = 3
    min_window_fill: int   = 3

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if self.hard_threshold <= 0:
            raise ValueError("hard_threshold must be positive.")
        if self.trend_steps < 2:
            raise ValueError("trend_steps must be at least 2.")
        if self.min_window_fill < 1 or self.min_window_fill > self.window_size:
            raise ValueError("min_window_fill must be between 1 and window_size.")


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntropyResult:
    """
    Output produced by EntropyMonitor.evaluate() for a single step.

    Attributes
    ----------
    step_index       : The step index of the most recent event evaluated.
    entropy          : H(t) computed over the sliding window. None if fewer
                       than min_window_fill steps are available.
    meltdown_detected: True if H(t) >= hard_threshold.
    early_warning    : True if entropy is trending upward for trend_steps
                       consecutive evaluations (precedes hard meltdown).
    reason           : MeltdownReason if meltdown_detected is True.
    window_tools     : The list of tool names in the evaluation window.
    unique_tool_count: Number of distinct tools in the evaluation window.
    """
    step_index        : int
    entropy           : Optional[float]
    meltdown_detected : bool              = False
    early_warning     : bool              = False
    reason            : Optional[MeltdownReason] = None
    window_tools      : List[str]         = field(default_factory=list)
    unique_tool_count : int               = 0

    @property
    def is_safe(self) -> bool:
        return not self.meltdown_detected and not self.early_warning


# ─────────────────────────────────────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(tool_names: Sequence[str]) -> float:
    """
    Compute Shannon entropy (base-2) of a distribution of tool names.

    Parameters
    ----------
    tool_names : Sequence of tool name strings (the window contents).

    Returns
    -------
    H : Shannon entropy in bits. 0.0 if the sequence has one unique element.

    Notes
    -----
    H(t) = -Σ p(x) * log₂(p(x))
    where p(x) = count(x) / len(tool_names)

    Upper bound: log₂(N_unique_tools) — grows with diversity.
    Lower bound: 0.0 — agent is using a single tool exclusively.
    """
    n = len(tool_names)
    if n == 0:
        return 0.0

    counts  = Counter(tool_names)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:                       # Guard against log(0)
            entropy -= p * math.log2(p)

    return entropy


# ─────────────────────────────────────────────────────────────────────────────
# Primary Monitor
# ─────────────────────────────────────────────────────────────────────────────

class EntropyMonitor:
    """
    Stateless sliding-window Shannon entropy monitor.

    The monitor is STATELESS by design: it takes the full trajectory
    (or a relevant window of it) as input each time, making it trivially
    testable and replayable without any hidden internal state.

    Trend detection requires a history of recent entropy values which the
    caller is expected to maintain and pass in. The ``SessionEntropyTracker``
    helper (below) wraps this into a convenient stateful interface.

    Usage
    -----
        config  = EntropyConfig()
        monitor = EntropyMonitor(config)
        result  = monitor.evaluate(trajectory, recent_entropy_history)
        if result.meltdown_detected:
            # trigger meltdown handling
    """

    def __init__(self, config: Optional[EntropyConfig] = None) -> None:
        self.config = config or EntropyConfig()

    def evaluate(
        self,
        trajectory: Sequence[StepEvent],
        entropy_history: Optional[List[float]] = None,
    ) -> EntropyResult:
        """
        Evaluate the current entropy state of the agent trajectory.

        Parameters
        ----------
        trajectory      : Full ordered sequence of StepEvents in the session.
        entropy_history : Optional list of the last ``trend_steps`` H(t) values,
                          maintained by the caller. Used for trend detection.
                          If None or shorter than trend_steps, trend detection
                          is skipped for this evaluation.

        Returns
        -------
        EntropyResult
        """
        cfg          = self.config
        step_index   = len(trajectory) - 1 if trajectory else 0

        # Extract the sliding window.
        window       = list(trajectory[-cfg.window_size:])
        window_tools = [e.tool_name for e in window]

        # Not enough data yet — return a safe, null result.
        if len(window) < cfg.min_window_fill:
            return EntropyResult(
                step_index=step_index,
                entropy=None,
                window_tools=window_tools,
                unique_tool_count=len(set(window_tools)),
            )

        h = _shannon_entropy(window_tools)

        # ── Hard meltdown threshold ───────────────────────────────────────────
        if h >= cfg.hard_threshold:
            return EntropyResult(
                step_index=step_index,
                entropy=h,
                meltdown_detected=True,
                early_warning=False,
                reason=MeltdownReason.HIGH_ENTROPY,
                window_tools=window_tools,
                unique_tool_count=len(set(window_tools)),
            )

        # ── Trend early warning ───────────────────────────────────────────────
        early_warning = False
        if entropy_history and len(entropy_history) >= cfg.trend_steps - 1:
            # Build the sequence of H values to check: history + current
            check_seq = list(entropy_history[-(cfg.trend_steps - 1):]) + [h]
            # Strictly increasing means each H is greater than the one before.
            early_warning = all(
                check_seq[i] < check_seq[i + 1]
                for i in range(len(check_seq) - 1)
            )

        return EntropyResult(
            step_index=step_index,
            entropy=h,
            meltdown_detected=False,
            early_warning=early_warning,
            reason=None,
            window_tools=window_tools,
            unique_tool_count=len(set(window_tools)),
        )

    def compute_entropy(self, tool_names: Sequence[str]) -> float:
        """
        Convenience method: compute raw Shannon entropy of a tool name sequence.
        Useful for benchmarking and unit tests without needing a full trajectory.
        """
        return _shannon_entropy(tool_names)


# ─────────────────────────────────────────────────────────────────────────────
# Stateful Tracker (convenience wrapper for session use)
# ─────────────────────────────────────────────────────────────────────────────

class SessionEntropyTracker:
    """
    Stateful wrapper around EntropyMonitor that maintains the entropy history
    buffer automatically. Designed for use inside the Sotis runtime.

    The runtime calls ``push_event(step_event)`` on every new step.
    The tracker accumulates the full trajectory and the rolling entropy history,
    then delegates to the stateless ``EntropyMonitor.evaluate()`` internally.

    Attributes
    ----------
    monitor        : The underlying stateless EntropyMonitor.
    trajectory     : Ordered list of all StepEvents seen this session.
    entropy_history: Rolling list of computed H(t) values (bounded to last 50).
    latest_result  : The EntropyResult from the most recent evaluation.
    """

    _MAX_HISTORY = 50   # Maximum retained entropy values in history buffer.

    def __init__(self, config: Optional[EntropyConfig] = None) -> None:
        self.monitor         = EntropyMonitor(config)
        self.trajectory      : List[StepEvent]   = []
        self.entropy_history : List[float]        = []
        self.latest_result   : Optional[EntropyResult] = None

    def push_event(self, event: StepEvent) -> EntropyResult:
        """
        Register a new StepEvent and return the current entropy evaluation.

        This is the primary hot-path method called on every agent step.
        """
        self.trajectory.append(event)

        result = self.monitor.evaluate(
            trajectory=self.trajectory,
            entropy_history=self.entropy_history,
        )

        # Update the rolling history only when a valid H(t) was computed.
        if result.entropy is not None:
            self.entropy_history.append(result.entropy)
            if len(self.entropy_history) > self._MAX_HISTORY:
                self.entropy_history.pop(0)

        self.latest_result = result
        return result

    def reset(self) -> None:
        """
        Clear the entropy history after a context reset event.
        The trajectory is retained (for logging purposes) but the history
        buffer is flushed so the new execution window starts clean.
        """
        self.trajectory.clear()
        self.entropy_history.clear()
        self.latest_result = None
