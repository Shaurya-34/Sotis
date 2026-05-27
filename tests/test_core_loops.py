"""
tests.test_core_loops
=====================
Comprehensive unit and performance tests for sotis.core.loops.

Test coverage:
    1.  No loop — healthy diverse trajectory is safe.
    2.  Exact loop fires at repeat_threshold — immediately triggers.
    3.  Loop fires within window, not globally — verifies window isolation.
    4.  Multiple looping tools — all captured, dominant identified.
    5.  Dominant loop is the one with the highest count.
    6.  LoopResult.dominant_tool_name convenience property.
    7.  Below repeat_threshold does NOT fire (e.g., 2 repeats when threshold=3).
    8.  Full window with uniform tool — no repeat_threshold reached (edge case).
    9.  Single step — cannot detect loop.
    10. Empty trajectory — safe result returned.
    11. Window isolation — loop outside the window is ignored.
    12. SessionLoopTracker: stateful push and O(1) window management.
    13. SessionLoopTracker: reset clears window.
    14. SessionLoopTracker: fingerprint_counts reflects active window.
    15. LoopDetector.check_single_tool() method.
    16. LoopConfig validation — invalid configs raise ValueError.
    17. Latency benchmark — push_event() must complete in < 2ms.
"""

from __future__ import annotations

import time
from typing import List

import pytest

from sotis.core.loops import LoopConfig, LoopDetector, LoopResult, SessionLoopTracker
from sotis.core.schemas import MeltdownReason, StepEvent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_event(tool: str, step: int, args: dict | None = None) -> StepEvent:
    return StepEvent(
        step_index=step,
        tool_name=tool,
        tool_args=args or {},
    )


def make_trajectory(tools: List[str], args_list: List[dict] | None = None) -> List[StepEvent]:
    """Build a trajectory from a list of tool names. Optional per-step args."""
    args_list = args_list or [{}] * len(tools)
    return [make_event(t, i, a) for i, (t, a) in enumerate(zip(tools, args_list))]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Healthy Trajectory — No Loop
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthyTrajectory:
    def test_diverse_tools_are_safe(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(["a", "b", "c", "d", "e", "f"])
        result     = detector.evaluate(trajectory)
        assert result.meltdown_detected is False
        assert result.is_safe is True
        assert result.looping_tools == []

    def test_alternating_two_tools_is_safe(self) -> None:
        # a, b, a, b, a, b — each appears 3× in a window of 6, exactly at threshold
        # BUT with default repeat_threshold=3, 3 repeats DOES fire.
        # Let's verify with a non-uniform case that stays below threshold.
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=4))
        trajectory = make_trajectory(["a", "b", "a", "b", "a", "b"])
        result     = detector.evaluate(trajectory)
        # 3 occurrences each, threshold is 4 → safe.
        assert result.meltdown_detected is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact Repeat Threshold Fires Immediately
# ─────────────────────────────────────────────────────────────────────────────

class TestRepeatThreshold:
    def test_three_identical_calls_triggers(self) -> None:
        """3 identical (tool, args) in a window of 6 → meltdown."""
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        trajectory = make_trajectory(["search", "read_file", "search", "edit", "search", "list"])
        result     = detector.evaluate(trajectory)

        assert result.meltdown_detected is True
        assert result.reason == MeltdownReason.TOOL_LOOP

    def test_two_identical_below_threshold_is_safe(self) -> None:
        """2 identical calls when threshold=3 → safe."""
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        trajectory = make_trajectory(["search", "read_file", "search", "edit", "write", "list"])
        result     = detector.evaluate(trajectory)

        assert result.meltdown_detected is False

    def test_threshold_of_two_fires_on_second_repeat(self) -> None:
        """With threshold=2, any tool appearing twice triggers immediately."""
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=2))
        trajectory = make_trajectory(["search", "search", "read_file"])
        result     = detector.evaluate(trajectory)

        assert result.meltdown_detected is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Window Isolation — Loop Outside Window Is Ignored
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowIsolation:
    def test_loop_outside_window_is_ignored(self) -> None:
        """
        First 3 steps are a tight loop of 'search' (would trigger).
        But they're outside the current 6-step window.
        The window contains 6 clean, diverse steps.
        """
        detector = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))

        # 3 loops + 6 clean steps = 9 total; window sees only last 6
        old_loops = make_trajectory(["search"] * 3)
        clean     = make_trajectory(["a", "b", "c", "d", "e", "f"])
        # Re-index the clean steps so step_index is monotonic.
        clean_reindexed = [
            StepEvent(step_index=3 + i, tool_name=e.tool_name, tool_args=e.tool_args)
            for i, e in enumerate(clean)
        ]
        full_trajectory = old_loops + clean_reindexed
        result          = detector.evaluate(full_trajectory)

        assert result.meltdown_detected is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multiple Looping Tools
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleLoops:
    def test_two_looping_tools_both_captured(self) -> None:
        """
        Window: [a, a, a, b, b, b] — both 'a' and 'b' loop 3× each.
        Both should appear in looping_tools.
        """
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        trajectory = make_trajectory(["a", "a", "a", "b", "b", "b"])
        result     = detector.evaluate(trajectory)

        assert result.meltdown_detected is True
        tool_names = {name for name, _ in result.looping_tools}
        assert "a" in tool_names
        assert "b" in tool_names


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Dominant Loop Identification
# ─────────────────────────────────────────────────────────────────────────────

class TestDominantLoop:
    def test_dominant_is_highest_count(self) -> None:
        """'a' appears 4×, 'b' appears 2× → dominant is 'a'."""
        detector   = LoopDetector(LoopConfig(window_size=6, repeat_threshold=2))
        trajectory = make_trajectory(["a", "a", "b", "a", "b", "a"])
        result     = detector.evaluate(trajectory)

        assert result.dominant_tool_name == "a"

    def test_dominant_tool_name_property(self) -> None:
        detector   = LoopDetector(LoopConfig(window_size=3, repeat_threshold=2))
        trajectory = make_trajectory(["x", "x", "y"])
        result     = detector.evaluate(trajectory)

        if result.meltdown_detected:
            assert result.dominant_tool_name is not None
        else:
            assert result.dominant_tool_name is None

    def test_no_loop_dominant_is_none(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(["a", "b", "c", "d"])
        result     = detector.evaluate(trajectory)
        assert result.dominant_loop is None
        assert result.dominant_tool_name is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Args Hash Isolation — Same Tool, Different Args → No Loop
# ─────────────────────────────────────────────────────────────────────────────

class TestArgHashIsolation:
    def test_same_tool_different_args_not_a_loop(self) -> None:
        """
        read_file with different file paths are NOT identical calls.
        The loop detector should treat these as distinct events.
        """
        detector = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        trajectory = make_trajectory(
            ["read_file"] * 6,
            args_list=[
                {"path": "file_a.py"},
                {"path": "file_b.py"},
                {"path": "file_c.py"},
                {"path": "file_d.py"},
                {"path": "file_e.py"},
                {"path": "file_f.py"},
            ],
        )
        result = detector.evaluate(trajectory)
        # Each call has a different args_hash → different fingerprints → no loop.
        assert result.meltdown_detected is False

    def test_same_tool_same_args_is_a_loop(self) -> None:
        """read_file with identical path 3× is a definitive loop."""
        detector = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        trajectory = make_trajectory(
            ["read_file"] * 6,
            args_list=[{"path": "same_file.py"}] * 6,
        )
        result = detector.evaluate(trajectory)
        assert result.meltdown_detected is True


# ─────────────────────────────────────────────────────────────────────────────
# 8-10. Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_trajectory_returns_safe(self) -> None:
        detector = LoopDetector()
        result   = detector.evaluate([])
        assert result.meltdown_detected is False
        assert result.is_safe is True

    def test_single_step_is_safe(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(["search"])
        result     = detector.evaluate(trajectory)
        assert result.meltdown_detected is False

    def test_two_steps_distinct_tools_safe(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(["search", "read_file"])
        result     = detector.evaluate(trajectory)
        assert result.meltdown_detected is False

    def test_window_fingerprints_populated(self) -> None:
        detector   = LoopDetector(LoopConfig(window_size=4))
        trajectory = make_trajectory(["a", "b", "c", "d", "e"])
        result     = detector.evaluate(trajectory)
        assert len(result.window_fingerprints) == 4   # Only window_size fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# 11-14. SessionLoopTracker Stateful Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionLoopTracker:
    def test_tracker_detects_loop_progressively(self) -> None:
        tracker = SessionLoopTracker(LoopConfig(window_size=6, repeat_threshold=3))
        tools   = ["search", "read", "search", "edit", "search", "write"]

        results = []
        for i, t in enumerate(tools):
            results.append(tracker.push_event(make_event(t, i)))

        # 'search' appears at steps 0, 2, 4 — all within the 6-step window.
        final = results[-1]
        assert final.meltdown_detected is True

    def test_tracker_reset_clears_window(self) -> None:
        tracker = SessionLoopTracker(LoopConfig(window_size=3, repeat_threshold=2))
        # Push a loop.
        for i in range(3):
            tracker.push_event(make_event("loop_tool", i))
        assert tracker.latest_result.meltdown_detected is True

        tracker.reset()
        assert len(tracker.window) == 0
        assert tracker.latest_result is None

    def test_fingerprint_counts_reflect_window(self) -> None:
        tracker = SessionLoopTracker(LoopConfig(window_size=4))
        for i in range(5):
            tracker.push_event(make_event("a" if i % 2 == 0 else "b", i))
        # Window of 4: [b(3), a(4), ...] — check counts are present.
        counts = tracker.fingerprint_counts
        assert len(counts) > 0
        assert all(v >= 1 for v in counts.values())

    def test_tracker_step_index_is_global(self) -> None:
        tracker = SessionLoopTracker()
        events  = [make_event("tool", i) for i in range(10)]
        result  = None
        for e in events:
            result = tracker.push_event(e)
        assert result.step_index == 9   # 10th push → global step 9


# ─────────────────────────────────────────────────────────────────────────────
# 15. check_single_tool() Diagnostic Method
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSingleTool:
    def test_counts_tool_in_window(self) -> None:
        detector   = LoopDetector(LoopConfig(window_size=6))
        trajectory = make_trajectory(["search", "search", "read", "search", "edit", "list"])
        count      = detector.check_single_tool("search", trajectory)
        assert count == 3

    def test_absent_tool_returns_zero(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(["a", "b", "c", "d"])
        count      = detector.check_single_tool("nonexistent_tool", trajectory)
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 16. LoopConfig Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopConfigValidation:
    def test_window_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            LoopConfig(window_size=1)

    def test_repeat_threshold_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="repeat_threshold"):
            LoopConfig(repeat_threshold=1)

    def test_threshold_exceeds_window_raises(self) -> None:
        with pytest.raises(ValueError, match="repeat_threshold cannot exceed"):
            LoopConfig(window_size=3, repeat_threshold=5)

    def test_valid_config_constructs_fine(self) -> None:
        cfg = LoopConfig(window_size=6, repeat_threshold=3)
        assert cfg.window_size == 6
        assert cfg.repeat_threshold == 3


# ─────────────────────────────────────────────────────────────────────────────
# 17. Latency Benchmark (< 2ms per call)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopDetectorLatency:
    """
    Performance test: LoopDetector.evaluate() must complete in < 2ms.
    push_event() on SessionLoopTracker is tested for O(1) efficiency.
    """

    def test_evaluate_latency_under_2ms(self) -> None:
        detector   = LoopDetector()
        trajectory = make_trajectory(
            ["read_file", "write_code", "search", "edit", "validate", "list"] * 10
        )
        n_calls = 1_000

        start     = time.perf_counter()
        for _ in range(n_calls):
            detector.evaluate(trajectory)
        elapsed_s = time.perf_counter() - start

        avg_ms = (elapsed_s / n_calls) * 1000
        print(f"\n  ✓ LoopDetector.evaluate() avg latency: {avg_ms:.4f} ms")

        assert avg_ms < 2.0, (
            f"LoopDetector.evaluate() avg latency {avg_ms:.4f}ms exceeds 2ms target."
        )

    def test_session_tracker_push_latency(self) -> None:
        tracker = SessionLoopTracker()
        n_calls = 10_000

        start   = time.perf_counter()
        for i in range(n_calls):
            tracker.push_event(make_event(f"tool_{i % 5}", i))
        elapsed_s = time.perf_counter() - start

        avg_ms = (elapsed_s / n_calls) * 1000
        print(f"\n  ✓ SessionLoopTracker.push_event() avg latency: {avg_ms:.4f} ms")

        assert avg_ms < 2.0, (
            f"SessionLoopTracker.push_event() avg latency {avg_ms:.4f}ms exceeds 2ms."
        )


# ─────────────────────────────────────────────────────────────────────────────
# WorkspaceDensityGuard Tests (Option B Extension)
# ─────────────────────────────────────────────────────────────────────────────

from sotis.core.loops import WorkspaceDensityGuard

class TestWorkspaceDensityGuard:
    def test_density_consecutive_edits_trigger(self) -> None:
        """Editing the same file 3 times consecutively without test changes triggers meltdown."""
        guard = WorkspaceDensityGuard(max_consecutive_edits=3)
        
        # Edit 1
        e1 = StepEvent(step_index=0, tool_name="write_workspace_file", tool_args={"file_path": "app.py"})
        assert guard.push_event(e1) is False
        
        # Edit 2
        e2 = StepEvent(step_index=1, tool_name="write_workspace_file", tool_args={"file_path": "app.py"})
        assert guard.push_event(e2) is False
        
        # Edit 3
        e3 = StepEvent(step_index=2, tool_name="write_workspace_file", tool_args={"file_path": "app.py"})
        assert guard.push_event(e3) is True
        assert guard.latest_meltdown is True
        assert guard.triggered_file == "app.py"

    def test_density_different_files_dont_trigger(self) -> None:
        """Editing different files does not accumulate on a single counter."""
        guard = WorkspaceDensityGuard(max_consecutive_edits=3)
        
        assert guard.push_event(StepEvent(step_index=0, tool_name="write_file", tool_args={"path": "a.py"})) is False
        assert guard.push_event(StepEvent(step_index=1, tool_name="write_file", tool_args={"path": "b.py"})) is False
        assert guard.push_event(StepEvent(step_index=2, tool_name="write_file", tool_args={"path": "a.py"})) is False
        assert guard.push_event(StepEvent(step_index=3, tool_name="write_file", tool_args={"path": "b.py"})) is False
        assert guard.latest_meltdown is False

    def test_density_outcome_shift_resets_counters(self) -> None:
        """If tests are run and the test result changes, the edit counter resets."""
        guard = WorkspaceDensityGuard(max_consecutive_edits=3)
        
        # 1. First edit
        assert guard.push_event(StepEvent(step_index=0, tool_name="write_file", tool_args={"path": "app.py"})) is False
        
        # 2. Run tests -> returns Failure A
        assert guard.push_event(StepEvent(step_index=1, tool_name="execute_tests", result_summary="Failure A")) is False
        
        # 3. Second edit
        assert guard.push_event(StepEvent(step_index=2, tool_name="write_file", tool_args={"path": "app.py"})) is False
        
        # 4. Run tests -> returns Failure B (different result summary!)
        assert guard.push_event(StepEvent(step_index=3, tool_name="execute_tests", result_summary="Failure B")) is False
        
        # Because outcome shifted, the edit counter for app.py is reset to 0!
        # 5. Third edit (consecutive but since outcomes shifted, does not trigger!)
        assert guard.push_event(StepEvent(step_index=4, tool_name="write_file", tool_args={"path": "app.py"})) is False
        assert guard.latest_meltdown is False

    def test_density_identical_test_outcomes_dont_reset(self) -> None:
        """If tests return the same result summary, the edit counter is NOT reset."""
        guard = WorkspaceDensityGuard(max_consecutive_edits=3)
        
        # 1. Edit 1
        assert guard.push_event(StepEvent(step_index=0, tool_name="write_file", tool_args={"path": "app.py"})) is False
        # 2. Run tests -> returns Failure A
        assert guard.push_event(StepEvent(step_index=1, tool_name="execute_tests", result_summary="Failure A")) is False
        
        # 3. Edit 2
        assert guard.push_event(StepEvent(step_index=2, tool_name="write_file", tool_args={"path": "app.py"})) is False
        # 4. Run tests -> returns Failure A again (same result!)
        assert guard.push_event(StepEvent(step_index=3, tool_name="execute_tests", result_summary="Failure A")) is False
        
        # 5. Edit 3 -> triggers meltdown because test outcomes did not change!
        assert guard.push_event(StepEvent(step_index=4, tool_name="write_file", tool_args={"path": "app.py"})) is True
        assert guard.latest_meltdown is True


class TestSemanticQueryLoop:
    def test_jaccard_similarity_loop_triggers(self) -> None:
        """3 consecutive search queries with high semantic similarity trigger a loop meltdown."""
        detector = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        
        # 3 similar search events
        e1 = StepEvent(step_index=0, tool_name="web_search", tool_args={"query": "room-temperature superconductor breakthroughs"})
        e2 = StepEvent(step_index=1, tool_name="web_search", tool_args={"query": "room temperature superconductor breakthroughs"})
        e3 = StepEvent(step_index=2, tool_name="web_search", tool_args={"query": "room-temperature superconductor breakthroughs news"})
        
        res = detector.evaluate([e1, e2, e3])
        assert res.meltdown_detected is True
        assert res.dominant_tool_name == "web_search"

    def test_jaccard_similarity_different_queries_no_trigger(self) -> None:
        """Distinct search queries do not trigger the Jaccard loop detector."""
        detector = LoopDetector(LoopConfig(window_size=6, repeat_threshold=3))
        
        e1 = StepEvent(step_index=0, tool_name="web_search", tool_args={"query": "room-temperature superconductor"})
        e2 = StepEvent(step_index=1, tool_name="web_search", tool_args={"query": "best apple pie recipe offline"})
        e3 = StepEvent(step_index=2, tool_name="web_search", tool_args={"query": "quantum computing companies 2026"})
        
        res = detector.evaluate([e1, e2, e3])
        assert res.meltdown_detected is False


