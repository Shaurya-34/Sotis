"""
tests/test_checkpoint
=====================
Unit tests for Git-style workspace checkpointing (checkpoint.py).
"""

import json
import os
from pathlib import Path
import pytest

from sotis.core.checkpoint import CheckpointManager, FileBaseline, FileDiff, WorkspaceCheckpoint
from sotis.core.schemas import ExecutionState, MeltdownSignal, MeltdownReason, Domain


def test_file_baseline_captured():
    """Test that FileBaseline captures correct state on construction."""
    baseline = FileBaseline(path="/mock/path.py", content="print('hello')", exists=True)
    assert baseline.path == "/mock/path.py"
    assert baseline.content == "print('hello')"
    assert baseline.exists is True
    assert baseline.captured_at > 0


def test_file_diff_properties():
    """Test the properties of FileDiff."""
    diff = FileDiff(
        path="/mock/path.py",
        status="modified",
        unified_diff="--- a/path.py\n+++ b/path.py\n@@ -1 +1 @@\n-hello\n+world",
        lines_added=1,
        lines_removed=1
    )
    assert diff.path == "/mock/path.py"
    assert diff.status == "modified"
    assert diff.has_changes is True
    assert diff.lines_added == 1
    assert diff.lines_removed == 1

    unchanged = FileDiff(path="/mock/path.py", status="unchanged")
    assert unchanged.has_changes is False


def test_workspace_checkpoint_json_roundtrip():
    """Test that a WorkspaceCheckpoint can serialize and deserialize cleanly to/from JSON."""
    file_diffs = {
        "/mock/path.py": FileDiff(
            path="/mock/path.py",
            status="modified",
            unified_diff="diff-text",
            lines_added=5,
            lines_removed=2
        ),
        "/mock/new.py": FileDiff(
            path="/mock/new.py",
            status="added",
            lines_added=10
        ),
        "/mock/deleted.py": FileDiff(
            path="/mock/deleted.py",
            status="deleted",
            lines_removed=8
        )
    }

    checkpoint = WorkspaceCheckpoint(
        session_id="session-123",
        subtask_id="subtask-abc",
        snapshot_at_step=5,
        snapshot_at_ms=123456789.0,
        completed_subtasks=["subtask-1"],
        active_subtask_desc="Active subtask description",
        trajectory_summary=[{"step": 1, "tool": "read_file", "args_hash": "hash123", "result": "ok"}],
        file_diffs=file_diffs,
        reset_number=1,
        meltdown_reason="HIGH_ENTROPY"
    )

    assert checkpoint.modified_files == ["/mock/path.py"]
    assert checkpoint.added_files == ["/mock/new.py"]
    assert checkpoint.deleted_files == ["/mock/deleted.py"]
    assert checkpoint.total_lines_changed == 25

    # Serialize
    serialized = checkpoint.to_json()
    assert isinstance(serialized, str)

    # Deserialize
    deserialized = WorkspaceCheckpoint.from_json(serialized)
    assert deserialized.session_id == checkpoint.session_id
    assert deserialized.subtask_id == checkpoint.subtask_id
    assert deserialized.snapshot_at_step == checkpoint.snapshot_at_step
    assert deserialized.snapshot_at_ms == checkpoint.snapshot_at_ms
    assert deserialized.completed_subtasks == checkpoint.completed_subtasks
    assert deserialized.active_subtask_desc == checkpoint.active_subtask_desc
    assert deserialized.trajectory_summary == checkpoint.trajectory_summary
    assert deserialized.reset_number == checkpoint.reset_number
    assert deserialized.meltdown_reason == checkpoint.meltdown_reason
    
    # Check that FileDiff instances are restored
    assert isinstance(deserialized.file_diffs["/mock/path.py"], FileDiff)
    assert deserialized.file_diffs["/mock/path.py"].status == "modified"
    assert deserialized.file_diffs["/mock/path.py"].lines_added == 5
    assert deserialized.file_diffs["/mock/new.py"].status == "added"
    assert deserialized.file_diffs["/mock/deleted.py"].status == "deleted"


def test_checkpoint_manager_track_and_untrack(tmp_path):
    """Test that CheckpointManager tracks and untracks files correctly."""
    mgr = CheckpointManager()
    file_a = tmp_path / "a.py"
    file_a.write_text("content-a", encoding="utf-8")

    file_b = tmp_path / "b.py" # doesn't exist yet

    # Track both files
    mgr.track([str(file_a), str(file_b)])

    assert len(mgr.tracked_paths) == 2
    assert str(file_a.resolve()) in mgr.tracked_paths
    assert str(file_b.resolve()) in mgr.tracked_paths

    # Check baselines
    baseline_a = mgr._baselines[str(file_a.resolve())]
    assert baseline_a.content == "content-a"
    assert baseline_a.exists is True

    baseline_b = mgr._baselines[str(file_b.resolve())]
    assert baseline_b.content == ""
    assert baseline_b.exists is False

    # Untrack
    mgr.untrack(str(file_a))
    assert len(mgr.tracked_paths) == 1
    assert str(file_a.resolve()) not in mgr.tracked_paths


def test_checkpoint_manager_snapshot_lifecycle(tmp_path):
    """Test generating unified diffs, addition, deletion, and unmodified detection on snapshot."""
    mgr = CheckpointManager()
    
    file_mod = tmp_path / "mod.py"
    file_mod.write_text("line1\nline2\nline3\n", encoding="utf-8")

    file_del = tmp_path / "del.py"
    file_del.write_text("to delete\n", encoding="utf-8")

    file_add = tmp_path / "add.py"  # does not exist initially

    file_same = tmp_path / "same.py"
    file_same.write_text("unchanged\n", encoding="utf-8")

    # Start tracking
    mgr.track([str(file_mod), str(file_del), str(file_add), str(file_same)])

    # Modify workspace
    file_mod.write_text("line1\nline2 edited\nline3\nline4 added\n", encoding="utf-8")
    os.remove(file_del)
    file_add.write_text("new file content\n", encoding="utf-8")
    # file_same is untouched

    # Setup state and signal
    state = ExecutionState(session_id="session-xyz", domain=Domain.SOFTWARE_ENGINEERING)
    signal = MeltdownSignal(
        session_id="session-xyz",
        subtask_id="sub-1",
        triggered_at_step=3,
        reason=MeltdownReason.HIGH_ENTROPY,
        entropy_value=1.8,
        reset_attempt=1
    )

    # Perform snapshot
    checkpoint = mgr.snapshot(state, signal)

    # Verify checkpoint contents
    assert checkpoint.session_id == "session-xyz"
    assert checkpoint.subtask_id == "sub-1"
    assert checkpoint.snapshot_at_step == 3
    assert checkpoint.reset_number == 1
    assert checkpoint.meltdown_reason == MeltdownReason.HIGH_ENTROPY.value

    # Verify diffs
    diffs = checkpoint.file_diffs
    
    # 1. Modified
    mod_diff = diffs[str(file_mod.resolve())]
    assert mod_diff.status == "modified"
    assert mod_diff.lines_added > 0
    assert mod_diff.lines_removed > 0
    assert "line2 edited" in mod_diff.unified_diff
    assert "+line4 added" in mod_diff.unified_diff

    # 2. Deleted
    del_diff = diffs[str(file_del.resolve())]
    assert del_diff.status == "deleted"
    assert del_diff.lines_removed == 1

    # 3. Added
    add_diff = diffs[str(file_add.resolve())]
    assert add_diff.status == "added"
    assert add_diff.lines_added == 1

    # 4. Unchanged
    same_diff = diffs[str(file_same.resolve())]
    assert same_diff.status == "unchanged"
    assert same_diff.unified_diff == ""

    # Verify check_manager state properties
    assert mgr.checkpoint_count == 1
    assert mgr.latest_checkpoint == checkpoint
    assert mgr.get_checkpoint(0) == checkpoint
    assert mgr.get_checkpoint(99) is None


def test_checkpoint_manager_reset_baselines(tmp_path):
    """Test reset_baselines updates the tracking baseline to current content."""
    mgr = CheckpointManager()
    file_a = tmp_path / "a.py"
    file_a.write_text("original content", encoding="utf-8")

    mgr.track([str(file_a)])
    
    # Modify file
    file_a.write_text("modified content", encoding="utf-8")

    # Reset baselines
    mgr.reset_baselines()

    # The new baseline should have "modified content"
    baseline = mgr._baselines[str(file_a.resolve())]
    assert baseline.content == "modified content"
    assert baseline.exists is True

    # Taking a snapshot now should return unchanged status
    state = ExecutionState(session_id="sess", domain=Domain.SOFTWARE_ENGINEERING)
    signal = MeltdownSignal(
        session_id="sess",
        subtask_id=None,
        triggered_at_step=0,
        reason=MeltdownReason.HIGH_ENTROPY,
        reset_attempt=1
    )
    checkpoint = mgr.snapshot(state, signal)
    assert checkpoint.file_diffs[str(file_a.resolve())].status == "unchanged"
