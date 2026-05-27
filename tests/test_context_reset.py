"""
tests/test_context_reset
========================
Unit tests for context distillation and prompt building (reset.py).
"""

from typing import List
import pytest

from sotis.core.checkpoint import FileDiff, WorkspaceCheckpoint
from sotis.core.reset import ContextResetter, DistillationConfig, DistillationResult, _estimate_tokens, _truncate_diff
from sotis.core.schemas import ExecutionState, StepEvent, MeltdownReason, Domain, Subtask


def test_distillation_config_defaults():
    """Test default values of DistillationConfig."""
    cfg = DistillationConfig()
    assert cfg.max_observations == 8
    assert cfg.include_file_diffs is True
    assert cfg.include_warning_note is True
    assert cfg.max_diff_lines_per_file == 30


def test_distillation_result_properties():
    """Test properties of DistillationResult."""
    res_pass = DistillationResult(
        prompt="prompt-content",
        raw_token_estimate=100,
        distilled_token_estimate=30,
        token_reduction_pct=70.0,
        reset_number=1,
        session_id="sess-1"
    )
    assert res_pass.meets_reduction_target is True
    assert "Token reduction: 70.0%" in res_pass.summary_line()
    assert "Target met: True" in res_pass.summary_line()

    res_fail = DistillationResult(
        prompt="prompt-content",
        raw_token_estimate=100,
        distilled_token_estimate=50,
        token_reduction_pct=50.0,
        reset_number=1,
        session_id="sess-1"
    )
    assert res_fail.meets_reduction_target is False
    assert "Target met: False" in res_fail.summary_line()


def test_estimate_tokens():
    """Test the chars/4 token estimation heuristic."""
    assert _estimate_tokens("1234") == 1
    assert _estimate_tokens("12345678") == 2
    assert _estimate_tokens("") == 1  # max(1, len // 4)


def test_truncate_diff():
    """Test that long diffs are correctly truncated with a descriptive note."""
    diff_text = "line1\nline2\nline3\nline4\nline5"
    
    # Under limit
    assert _truncate_diff(diff_text, 10) == diff_text

    # Over limit
    truncated = _truncate_diff(diff_text, 3)
    lines = truncated.splitlines()
    assert len(lines) == 4
    assert lines[0] == "line1"
    assert lines[2] == "line3"
    assert "... [2 more lines truncated] ..." in lines[3]


def test_context_resetter_distill_basic():
    """Test basic context distillation with progress, subtasks, observations, and diffs."""
    state = ExecutionState(session_id="session-1", domain=Domain.SOFTWARE_ENGINEERING)
    
    # Setup subtasks
    state.subtasks = [
        Subtask(subtask_id="sub-1", description="Implement feature", status="DONE", gds_weight=0.5),
        Subtask(subtask_id="sub-2", description="Test feature", status="ACTIVE", gds_weight=0.5)
    ]
    state.active_subtask_id = "sub-2"

    # Add trajectory steps
    state.record_step(StepEvent(step_index=0, tool_name="read_file", tool_args={"path": "a.py"}, result_summary="a content", subtask_id="sub-1"))
    state.record_step(StepEvent(step_index=1, tool_name="write_file", tool_args={"path": "a.py"}, result_summary="ok", subtask_id="sub-1"))
    state.record_step(StepEvent(step_index=2, tool_name="run_tests", tool_args={"cmd": "pytest"}, result_summary="test failed", subtask_id="sub-2"))
    state.record_step(StepEvent(step_index=3, tool_name="run_tests", tool_args={"cmd": "pytest"}, result_summary="test failed again", subtask_id="sub-2"))

    # Setup Checkpoint
    file_diffs = {
        "a.py": FileDiff(path="a.py", status="modified", unified_diff="--- a.py\n+++ a.py\n+new line", lines_added=1, lines_removed=0)
    }
    checkpoint = WorkspaceCheckpoint(
        session_id="session-1",
        subtask_id="sub-2",
        snapshot_at_step=3,
        snapshot_at_ms=1000.0,
        completed_subtasks=["sub-1"],
        active_subtask_desc="Test feature",
        trajectory_summary=[],
        file_diffs=file_diffs,
        reset_number=1,
        meltdown_reason="TOOL_LOOP"
    )

    resetter = ContextResetter()
    result = resetter.distill(state, checkpoint, task_goal="Build a working calculator")

    # Assertions on token reduction calculation
    assert result.session_id == "session-1"
    assert result.reset_number == 1
    assert result.raw_token_estimate > 0
    assert result.distilled_token_estimate > 0
    assert isinstance(result.token_reduction_pct, float)

    # Assertions on prompt contents
    prompt = result.prompt
    assert "Context Reset Notice (Reset #1/2)" in prompt
    assert "repeated identical tool calls" in prompt
    assert "Build a working calculator" in prompt
    assert "Verified Progress" in prompt
    assert "[DONE] Implement feature  (GDS weight: 50%)" in prompt
    assert "Current Subtask (Resume Here)" in prompt
    assert "Goal:         Test feature" in prompt
    assert "Key Observations from Previous Execution" in prompt
    assert "[run_tests] → test failed again" in prompt
    assert "[write_file] → ok" in prompt
    assert "[read_file] → a content" in prompt
    assert "Workspace State at Reset Point" in prompt
    assert "Modified files (1):" in prompt
    assert "~ a.py" in prompt
    assert "Instructions" in prompt


def test_context_resetter_distill_config_toggles():
    """Test context resetter honors toggles in DistillationConfig."""
    state = ExecutionState(session_id="session-1", domain=Domain.SOFTWARE_ENGINEERING)
    checkpoint = WorkspaceCheckpoint(
        session_id="session-1",
        subtask_id=None,
        snapshot_at_step=2,
        snapshot_at_ms=1000.0,
        completed_subtasks=[],
        active_subtask_desc="",
        trajectory_summary=[],
        file_diffs={},
        reset_number=1,
        meltdown_reason="HIGH_ENTROPY"
    )

    # 1. Warning note disabled
    cfg1 = DistillationConfig(include_warning_note=False)
    resetter1 = ContextResetter(cfg1)
    res1 = resetter1.distill(state, checkpoint, task_goal="Goal")
    assert "Context Reset Notice" not in res1.prompt

    # 2. File diffs disabled
    file_diffs = {"a.py": FileDiff(path="a.py", status="modified", unified_diff="some-diff")}
    checkpoint_with_diff = WorkspaceCheckpoint(
        session_id="session-1",
        subtask_id=None,
        snapshot_at_step=2,
        snapshot_at_ms=1000.0,
        completed_subtasks=[],
        active_subtask_desc="",
        trajectory_summary=[],
        file_diffs=file_diffs,
        reset_number=1,
        meltdown_reason="HIGH_ENTROPY"
    )
    cfg2 = DistillationConfig(include_file_diffs=False)
    resetter2 = ContextResetter(cfg2)
    res2 = resetter2.distill(state, checkpoint_with_diff, task_goal="Goal")
    assert "Workspace State at Reset Point" not in res2.prompt


def test_context_resetter_observations_limiting():
    """Test observations collection matches max_observations and reverses order."""
    state = ExecutionState(session_id="session-1")
    
    # 5 unique tools
    state.record_step(StepEvent(step_index=0, tool_name="t1", result_summary="res1"))
    state.record_step(StepEvent(step_index=1, tool_name="t2", result_summary="res2"))
    state.record_step(StepEvent(step_index=2, tool_name="t3", result_summary="res3"))
    state.record_step(StepEvent(step_index=3, tool_name="t4", result_summary="res4"))
    state.record_step(StepEvent(step_index=4, tool_name="t5", result_summary="res5"))

    checkpoint = WorkspaceCheckpoint(
        session_id="session-1",
        subtask_id=None,
        snapshot_at_step=4,
        snapshot_at_ms=1000.0,
        completed_subtasks=[],
        active_subtask_desc="",
        trajectory_summary=[],
        file_diffs={},
        reset_number=1,
        meltdown_reason="HIGH_ENTROPY"
    )

    # Max observations = 3, should only contain the 3 most recent unique tools (t5, t4, t3)
    cfg = DistillationConfig(max_observations=3)
    resetter = ContextResetter(cfg)
    res = resetter.distill(state, checkpoint, task_goal="Goal")

    assert "[t5]" in res.prompt
    assert "[t4]" in res.prompt
    assert "[t3]" in res.prompt
    assert "[t2]" not in res.prompt
    assert "[t1]" not in res.prompt


def test_context_resetter_token_reduction_target_met():
    """Verify that under typical large trajectories Sotis achieves >= 60% token reduction."""
    state = ExecutionState(session_id="session-1", domain=Domain.SOFTWARE_ENGINEERING)
    
    # Setup subtasks
    state.subtasks = [
        Subtask(subtask_id="sub-1", description="Decompose", status="DONE", gds_weight=0.5),
        Subtask(subtask_id="sub-2", description="Work", status="ACTIVE", gds_weight=0.5)
    ]
    state.active_subtask_id = "sub-2"

    # Inject 150 steps with extremely verbose results in the raw trajectory to simulate a heavy run
    for i in range(150):
        state.record_step(StepEvent(
            step_index=i,
            tool_name=f"tool_{i}",
            tool_args={"some_arg": "value_that_is_quite_long_and_verbose_to_inflate_trajectory"},
            result_summary="x" * 450,  # 450 chars summary per step
            subtask_id="sub-2"
        ))

    checkpoint = WorkspaceCheckpoint(
        session_id="session-1",
        subtask_id="sub-2",
        snapshot_at_step=149,
        snapshot_at_ms=1000.0,
        completed_subtasks=["sub-1"],
        active_subtask_desc="Work",
        trajectory_summary=[],
        file_diffs={},
        reset_number=1,
        meltdown_reason="HIGH_ENTROPY"
    )

    resetter = ContextResetter()
    result = resetter.distill(state, checkpoint, task_goal="Task Goal")

    assert result.raw_token_estimate > 4000
    assert result.token_reduction_pct >= 60.0
    assert result.meets_reduction_target is True

