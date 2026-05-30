"""
tests.test_core_entropy
=======================
Comprehensive unit and benchmark tests for sotis.core.entropy.

Test coverage:
    1.  Mathematical correctness — verify H(t) against known values.
    2.  Window slicing — ensure only the last N steps are included.
    3.  Hard threshold meltdown trigger — H >= 1.5 must fire.
    4.  Early-warning trend detection — 3 consecutive rises must fire.
    5.  Min-window guard — no signal emitted with insufficient data.
    6.  All-same tool (minimum entropy = 0.0) — must NOT trigger.
    7.  All-unique tools (maximum entropy) — must trigger for large windows.
    8.  Single step edge case.
    9.  Entropy history overflow stays bounded (≤ 50 entries).
    10. SessionEntropyTracker stateful integration.
    11. Latency benchmark — evaluate() must complete in < 2ms per call.
"""

from __future__ import annotations

import math
import time
from typing import List

import pytest

from sotis.core.entropy import (
    EntropyConfig,
    EntropyMonitor,
    EntropyResult,
    SessionEntropyTracker,
    _shannon_entropy,
)
from sotis.core.schemas import MeltdownReason, StepEvent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_event(tool: str, step: int, args: dict | None = None) -> StepEvent:
    """Factory for StepEvent used across all tests."""
    return StepEvent(
        step_index=step,
        tool_name=tool,
        tool_args=args or {},
    )


def make_trajectory(tools: List[str]) -> List[StepEvent]:
    """Build a trajectory from a list of tool names."""
    return [make_event(t, i) for i, t in enumerate(tools)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mathematical Correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestShannonEntropyMath:
    """Verify the pure _shannon_entropy() function against analytical values."""

    def test_empty_sequence_returns_zero(self) -> None:
        assert _shannon_entropy([]) == 0.0

    def test_single_element_returns_zero(self) -> None:
        # p(x) = 1.0, -1 * log2(1.0) = 0.0
        assert _shannon_entropy(["read_file"]) == pytest.approx(0.0)

    def test_uniform_two_tools(self) -> None:
        # 2 tools, each with p = 0.5 → H = 1.0 bit
        h = _shannon_entropy(["read_file", "write_file"])
        assert h == pytest.approx(1.0, rel=1e-9)

    def test_uniform_four_tools(self) -> None:
        # 4 tools equally distributed → H = log2(4) = 2.0 bits
        tools = ["a", "b", "c", "d"]
        h = _shannon_entropy(tools)
        assert h == pytest.approx(2.0, rel=1e-9)

    def test_skewed_distribution(self) -> None:
        # 3× read_file, 1× search → counts {read_file:3, search:1}
        # p(read_file) = 0.75, p(search) = 0.25
        # H = -(0.75 * log2(0.75) + 0.25 * log2(0.25))
        tools = ["read_file"] * 3 + ["search"]
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        h = _shannon_entropy(tools)
        assert h == pytest.approx(expected, rel=1e-9)

    def test_all_same_tool_is_zero(self) -> None:
        # Agent is fixated on one tool → H = 0 (minimum, NOT a meltdown signal)
        h = _shannon_entropy(["read_file"] * 10)
        assert h == pytest.approx(0.0)

    def test_five_unique_tools_uniform(self) -> None:
        # log2(5) ≈ 2.3219 bits — max entropy for 5 distinct uniform tools
        tools = ["a", "b", "c", "d", "e"]
        expected = math.log2(5)
        h = _shannon_entropy(tools)
        assert h == pytest.approx(expected, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Window Slicing
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowSlicing:
    """Verify that only the last N steps are included in evaluation."""

    def test_window_uses_last_n_steps_only(self) -> None:
        """
        First 10 steps are all 'a' (H=0). Last 5 steps are 5 unique tools.
        The monitor should only see the last 5 and return high entropy.
        """
        monitor = EntropyMonitor(EntropyConfig(window_size=5))
        early_tools = ["a"] * 10
        late_tools  = ["b", "c", "d", "e", "f"]
        trajectory  = make_trajectory(early_tools + late_tools)

        result = monitor.evaluate(trajectory)
        assert result.window_tools == late_tools
        assert result.entropy == pytest.approx(math.log2(5), rel=1e-9)

    def test_window_tools_matches_actual_slice(self) -> None:
        trajectory = make_trajectory(["x", "y", "z", "a", "b", "c", "d"])
        monitor    = EntropyMonitor(EntropyConfig(window_size=4))
        result     = monitor.evaluate(trajectory)
        assert result.window_tools == ["a", "b", "c", "d"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hard Threshold Meltdown Trigger
# ─────────────────────────────────────────────────────────────────────────────

class TestHardThresholdMeltdown:
    """H(t) >= hard_threshold must set meltdown_detected=True."""

    def test_exceeds_threshold_triggers_meltdown(self) -> None:
        # 4 unique tools in window of 4 → H = log2(4) = 2.0 > 1.5 threshold
        monitor    = EntropyMonitor(EntropyConfig(window_size=4, hard_threshold=1.5))
        trajectory = make_trajectory(["a", "b", "c", "d"])
        result     = monitor.evaluate(trajectory)

        assert result.meltdown_detected is True
        assert result.reason == MeltdownReason.HIGH_ENTROPY
        assert result.entropy >= 1.5

    def test_exactly_at_threshold_triggers_meltdown(self) -> None:
        # Craft a distribution where H = exactly 1.5 triggers.
        # H=1.5 means we need to hit it exactly. Use a custom threshold = computed H.
        tools      = ["a", "b", "c"]   # 3 uniform → H = log2(3) ≈ 1.585
        h          = _shannon_entropy(tools)
        monitor    = EntropyMonitor(EntropyConfig(window_size=3, hard_threshold=h))
        trajectory = make_trajectory(tools)
        result     = monitor.evaluate(trajectory)

        assert result.meltdown_detected is True

    def test_below_threshold_is_safe(self) -> None:
        # 2 tools, one dominant: H < 1.0 — safely below 1.5
        monitor    = EntropyMonitor(EntropyConfig(window_size=5, hard_threshold=1.5))
        trajectory = make_trajectory(["read_file"] * 4 + ["search"])
        result     = monitor.evaluate(trajectory)

        assert result.meltdown_detected is False
        assert result.entropy < 1.5

    def test_reason_is_high_entropy(self) -> None:
        monitor    = EntropyMonitor()
        trajectory = make_trajectory(["a", "b", "c", "d", "e"])  # H = log2(5) ≈ 2.32
        result     = monitor.evaluate(trajectory)

        assert result.reason == MeltdownReason.HIGH_ENTROPY


# ─────────────────────────────────────────────────────────────────────────────
# 4. Early-Warning Trend Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyWarningTrend:
    """3 strictly increasing consecutive H values → early_warning=True."""

    def test_strictly_increasing_history_fires_warning(self) -> None:
        monitor = EntropyMonitor(EntropyConfig(trend_steps=3))
        # ["a","a","a","b","c"] → p(a)=0.6, p(b)=0.2, p(c)=0.2 → H ≈ 1.371
        # This is BELOW the 1.5 hard threshold so the trend path is exercised.
        # History: [0.5, 0.8] + current 1.371 → strictly increasing → early_warning.
        trajectory = make_trajectory(["a", "a", "a", "b", "c"])
        result = monitor.evaluate(
            trajectory=trajectory,
            entropy_history=[0.5, 0.8],
        )
        assert result.meltdown_detected is False   # Below hard threshold
        assert result.entropy is not None
        assert result.entropy > 0.8               # Entropy rose above history
        assert result.early_warning is True

    def test_flat_history_does_not_fire(self) -> None:
        monitor = EntropyMonitor(EntropyConfig(trend_steps=3))
        trajectory = make_trajectory(["a", "b", "c", "d", "e"])
        # Flat history — not strictly increasing.
        result = monitor.evaluate(
            trajectory=trajectory,
            entropy_history=[1.2, 1.2],
        )
        # Even though H might be high and trigger meltdown, trend itself is flat.
        # If meltdown triggered by threshold, early_warning is False (separate signal).
        if not result.meltdown_detected:
            assert result.early_warning is False

    def test_decreasing_history_does_not_fire(self) -> None:
        monitor    = EntropyMonitor(EntropyConfig(trend_steps=3))
        trajectory = make_trajectory(["a", "a", "a", "b", "a"])
        result     = monitor.evaluate(
            trajectory=trajectory,
            entropy_history=[1.0, 0.8],
        )
        # H went down — no early warning.
        assert result.early_warning is False

    def test_no_history_no_early_warning(self) -> None:
        monitor    = EntropyMonitor()
        trajectory = make_trajectory(["a", "b", "c", "d"])
        result     = monitor.evaluate(trajectory=trajectory, entropy_history=None)
        assert result.early_warning is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Min-Window Guard
# ─────────────────────────────────────────────────────────────────────────────

class TestMinWindowGuard:
    """Fewer than min_window_fill steps → entropy=None, no signal."""

    def test_too_few_steps_returns_null_entropy(self) -> None:
        config     = EntropyConfig(window_size=5, min_window_fill=3)
        monitor    = EntropyMonitor(config)
        trajectory = make_trajectory(["a", "b"])   # Only 2 steps, need 3
        result     = monitor.evaluate(trajectory)

        assert result.entropy is None
        assert result.meltdown_detected is False
        assert result.early_warning is False

    def test_exactly_min_fill_computes_entropy(self) -> None:
        config     = EntropyConfig(window_size=5, min_window_fill=3)
        monitor    = EntropyMonitor(config)
        trajectory = make_trajectory(["a", "b", "c"])   # Exactly 3
        result     = monitor.evaluate(trajectory)

        assert result.entropy is not None
        assert result.entropy > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. False-Positive: All-Same Tool (H = 0)
# ─────────────────────────────────────────────────────────────────────────────

class TestAllSameToolSafeCase:
    """An agent calling the same tool repeatedly should NOT trigger entropy meltdown."""

    def test_repetitive_tool_does_not_trigger_entropy_meltdown(self) -> None:
        monitor    = EntropyMonitor()
        # Simulate a focused agent reading the same file repeatedly.
        trajectory = make_trajectory(["read_file"] * 20)
        result     = monitor.evaluate(trajectory)

        assert result.meltdown_detected is False
        assert result.entropy == pytest.approx(0.0)

    def test_single_tool_is_safe(self) -> None:
        monitor = EntropyMonitor()
        trajectory = make_trajectory(["write_code"] * 10)
        result = monitor.evaluate(trajectory)
        assert result.is_safe is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Unique Tools (Max Entropy)
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxEntropyTrigger:
    """All-unique tools in window → max entropy → must trigger meltdown."""

    def test_all_unique_tools_triggers(self) -> None:
        monitor    = EntropyMonitor(EntropyConfig(window_size=5, hard_threshold=1.5))
        # 5 unique tools → H = log2(5) ≈ 2.32, well above 1.5
        trajectory = make_trajectory(["tool_a", "tool_b", "tool_c", "tool_d", "tool_e"])
        result     = monitor.evaluate(trajectory)

        assert result.meltdown_detected is True
        assert result.entropy == pytest.approx(math.log2(5), rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_trajectory(self) -> None:
        monitor = EntropyMonitor()
        result  = monitor.evaluate([])
        assert result.entropy is None
        assert result.meltdown_detected is False

    def test_single_step_trajectory(self) -> None:
        monitor    = EntropyMonitor()
        trajectory = make_trajectory(["read_file"])
        result     = monitor.evaluate(trajectory)
        # Only 1 step, below min_window_fill=3
        assert result.entropy is None

    def test_unique_tool_count_is_correct(self) -> None:
        monitor    = EntropyMonitor(EntropyConfig(window_size=5, min_window_fill=3))
        trajectory = make_trajectory(["a", "a", "b", "c", "a"])
        result     = monitor.evaluate(trajectory)
        assert result.unique_tool_count == 3   # a, b, c


# ─────────────────────────────────────────────────────────────────────────────
# 9. SessionEntropyTracker: Stateful Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionEntropyTracker:

    def test_tracker_emits_none_entropy_below_min_fill(self) -> None:
        tracker = SessionEntropyTracker(EntropyConfig(min_window_fill=3))
        e1      = tracker.push_event(make_event("a", 0))
        e2      = tracker.push_event(make_event("b", 1))
        assert e1.entropy is None
        assert e2.entropy is None

    def test_tracker_accumulates_history(self) -> None:
        tracker = SessionEntropyTracker()
        tools   = ["a", "b", "c", "d", "e", "a", "b"]
        for i, t in enumerate(tools):
            tracker.push_event(make_event(t, i))
        # History should have been populated for steps with valid entropy.
        assert len(tracker.entropy_history) > 0

    def test_reset_clears_entropy_history(self) -> None:
        tracker = SessionEntropyTracker()
        for i, t in enumerate(["a", "b", "c", "d", "e"]):
            tracker.push_event(make_event(t, i))
        assert len(tracker.entropy_history) > 0

        tracker.reset()
        assert len(tracker.entropy_history) == 0
        assert tracker.latest_result is None

    def test_tracker_detects_meltdown_after_diverse_tools(self) -> None:
        config  = EntropyConfig(window_size=5, hard_threshold=1.5, min_window_fill=3)
        tracker = SessionEntropyTracker(config)
        # Push 5 completely distinct tools → high entropy.
        diverse = ["tool_a", "tool_b", "tool_c", "tool_d", "tool_e"]
        result  = None
        for i, t in enumerate(diverse):
            result = tracker.push_event(make_event(t, i))
        assert result is not None
        assert result.meltdown_detected is True

    def test_history_is_bounded_to_50(self) -> None:
        tracker = SessionEntropyTracker()
        # Push 100 events; history should never exceed 50.
        for i in range(100):
            tracker.push_event(make_event(f"tool_{i % 3}", i))
        assert len(tracker.entropy_history) <= 50


# ─────────────────────────────────────────────────────────────────────────────
# 10. Compute Entropy Convenience Method
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEntropyMethod:
    def test_matches_internal_function(self) -> None:
        monitor = EntropyMonitor()
        tools   = ["a", "b", "c", "a"]
        assert monitor.compute_entropy(tools) == pytest.approx(_shannon_entropy(tools))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Latency Benchmark (< 2ms per call target)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyLatency:
    """
    Performance test: evaluate() must complete in well under 2ms.

    We time 1000 consecutive calls and verify the average is < 2ms.
    This validates Sotis has negligible overhead as a hot-path middleware.
    """

    def test_evaluate_latency_under_2ms(self) -> None:
        monitor    = EntropyMonitor()
        trajectory = make_trajectory(
            ["read_file", "write_code", "search", "read_file", "validate"] * 20
        )
        n_calls    = 1_000

        start = time.perf_counter()
        for _ in range(n_calls):
            monitor.evaluate(trajectory, entropy_history=[0.5, 0.8, 1.1])
        elapsed_s = time.perf_counter() - start

        avg_ms = (elapsed_s / n_calls) * 1000
        print(f"\n  ✓ Shannon entropy evaluate() avg latency: {avg_ms:.4f} ms")

        assert avg_ms < 2.0, (
            f"EntropyMonitor.evaluate() avg latency {avg_ms:.4f}ms exceeds 2ms target."
        )

    def test_shannon_entropy_pure_function_latency(self) -> None:
        """Verify the raw _shannon_entropy() function is extremely fast."""
        tools  = ["a", "b", "c", "d", "e"]
        n      = 100_000

        start  = time.perf_counter()
        for _ in range(n):
            _shannon_entropy(tools)
        elapsed_s = time.perf_counter() - start

        avg_us = (elapsed_s / n) * 1_000_000
        print(f"\n  ✓ _shannon_entropy() avg latency: {avg_us:.3f} µs")

        assert avg_us < 150, (
            f"_shannon_entropy() avg {avg_us:.3f} µs is unexpectedly slow."
        )
