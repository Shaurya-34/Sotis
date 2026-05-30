"""
tests/test_runtime
==================
Integration and unit tests for the custom ReAct runtime (runtime.py).
"""

import os
from pathlib import Path
import pytest

from sotis.core.entropy import EntropyConfig
from sotis.core.loops import LoopConfig
from sotis.core.reset import DistillationConfig
from sotis.core.schemas import Domain, SessionStatus, MeltdownReason
from sotis.lib.adapters import MockAdapter, LLMResponse, ToolCall, LLMMessage
from sotis.lib.runtime import SotisRuntime, ToolDefinition, RunResult


# ─────────────────────────────────────────────────────────────────────────────
# Test Tools
# ─────────────────────────────────────────────────────────────────────────────

def mock_write_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return "File written successfully."


def mock_read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_runtime_basic_success():
    """Verify runtime completes normally when adapter returns a terminal response."""
    tool_defs = [
        ToolDefinition("write_file", "Write to a file", {"path": {"type": "string"}, "content": {"type": "string"}})
    ]
    tools = {"write_file": mock_write_file}

    # Queue contains: 
    # 1. Tool call to write_file
    # 2. Terminal response
    adapter = MockAdapter(responses=[
        LLMResponse(
            text="",
            tool_calls=[ToolCall("write_file", {"path": "a.txt", "content": "hello"})]
        ),
        LLMResponse(text="Task complete. I wrote the file.", tool_calls=[])
    ])

    runtime = SotisRuntime(
        adapter=adapter,
        tools=tools,
        tool_definitions=tool_defs,
        task_goal="Write 'hello' to a.txt",
        domain=Domain.SOFTWARE_ENGINEERING
    )

    result = runtime.run()

    assert result.success is True
    assert result.hard_failed is False
    assert result.total_steps == 1
    assert result.total_resets == 0
    assert result.final_response == "Task complete. I wrote the file."
    assert len(result.distillation_results) == 0
    assert result.state.status == SessionStatus.COMPLETED


def test_runtime_entropy_meltdown_and_recovery(tmp_path):
    """Test that an entropy meltdown triggers context reset and recovers successfully."""
    # Setup files to track
    tracked_file = tmp_path / "code.py"
    tracked_file.write_text("initial", encoding="utf-8")

    # Define tools and mock registry
    tool_defs = [
        ToolDefinition("tool_a", "Tool A", {}),
        ToolDefinition("tool_b", "Tool B", {}),
        ToolDefinition("tool_c", "Tool C", {}),
        ToolDefinition("write_file", "Write file", {"path": {"type": "string"}, "content": {"type": "string"}})
    ]
    
    # Tool functions
    executed_tools = []
    def tool_fn_a():
        executed_tools.append("a")
        return "a-ok"
    
    def tool_fn_b():
        executed_tools.append("b")
        return "b-ok"

    def tool_fn_c():
        executed_tools.append("c")
        return "c-ok"

    tools = {
        "tool_a": tool_fn_a,
        "tool_b": tool_fn_b,
        "tool_c": tool_fn_c,
        "write_file": mock_write_file
    }

    # To trigger entropy meltdown quickly:
    # Use min_window_fill=3, hard_threshold=1.0. 3 diverse tool calls (a, b, c) will produce H = log2(3) = 1.58 > 1.0.
    entropy_config = EntropyConfig(min_window_fill=3, hard_threshold=1.0)

    # Responses Queue:
    # First Context Attempt:
    # 1. Call tool_a
    # 2. Call tool_b
    # 3. Call tool_c -> triggers entropy meltdown! Sotis resets and resumes.
    # Second Context Attempt (after reset):
    # Sotis will inject distilled prompt as new user message.
    # 4. LLM should complete normally by writing file and finishing.
    adapter = MockAdapter(responses=[
        LLMResponse(text="", tool_calls=[ToolCall("tool_a", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("tool_b", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("tool_c", {})]),
        # --- RESET FIRES HERE ---
        LLMResponse(text="", tool_calls=[ToolCall("write_file", {"path": str(tracked_file), "content": "reset_write"})]),
        LLMResponse(text="All done after reset", tool_calls=[])
    ])

    runtime = SotisRuntime(
        adapter=adapter,
        tools=tools,
        tool_definitions=tool_defs,
        task_goal="Complete the diverse tasks and write code.py",
        workspace_paths=[str(tracked_file)],
        domain=Domain.SOFTWARE_ENGINEERING,
        entropy_config=entropy_config
    )

    result = runtime.run()

    assert result.success is True
    assert result.hard_failed is False
    assert result.total_resets == 1
    assert result.total_steps == 4  # 3 before reset + 1 after reset
    assert result.final_response == "All done after reset"
    assert len(result.distillation_results) == 1
    
    # Check distillation metrics
    distill = result.distillation_results[0]
    assert distill.reset_number == 1
    assert "Context Reset Notice" in distill.prompt
    assert "entropy" in distill.prompt.lower()
    
    # Check that tracked file shows up as modified in the second run checkpoint if modified later
    # (Here, we wrote "reset_write" after the reset baseline reset, so the next checkpoint wasn't taken,
    # but the baseline was reset correctly). Let's verify the file actually got updated.
    assert tracked_file.read_text(encoding="utf-8") == "reset_write"


def test_runtime_loop_meltdown_and_recovery():
    """Test that a loop meltdown (repeated tool calls) triggers context reset and recovers."""
    tool_defs = [
        ToolDefinition("read_file", "Read file", {"path": {"type": "string"}})
    ]
    
    call_count = 0
    def mock_read(path: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Content of {path}"

    tools = {"read_file": mock_read}

    # Loop configuration: window_size=4, repeat_threshold=2.
    # Repeats trigger loop detection when same (tool, args) pair is called 2 times in the window.
    loop_config = LoopConfig(window_size=4, repeat_threshold=2)

    # Responses Queue:
    # First Context Attempt:
    # 1. Call read_file("a.txt")
    # 2. Call read_file("a.txt") -> triggers loop meltdown! Sotis resets and resumes.
    # Second Context Attempt (after reset):
    # 3. LLM returns terminal response.
    adapter = MockAdapter(responses=[
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "a.txt"})]),
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "a.txt"})]),
        # --- RESET FIRES HERE ---
        LLMResponse(text="Recovered from loop", tool_calls=[])
    ])

    runtime = SotisRuntime(
        adapter=adapter,
        tools=tools,
        tool_definitions=tool_defs,
        task_goal="Read file",
        domain=Domain.SOFTWARE_ENGINEERING,
        loop_config=loop_config
    )

    result = runtime.run()

    assert result.success is True
    assert result.hard_failed is False
    assert result.total_resets == 1
    assert result.total_steps == 2
    assert result.final_response == "Recovered from loop"
    assert len(result.distillation_results) == 1
    
    distill = result.distillation_results[0]
    assert "loop" in distill.prompt.lower()
    assert "read_file" in distill.prompt


def test_runtime_hard_fail_enforcement():
    """Verify Sotis hard-fails and exits once 2 context resets are exhausted for a subtask."""
    tool_defs = [
        ToolDefinition("loop_tool", "Loops forever", {})
    ]
    tools = {"loop_tool": lambda: "output"}

    # Trigger loop meltdown quickly
    loop_config = LoopConfig(window_size=4, repeat_threshold=2)

    # Setup an adapter that ALWAYS returns a loop_tool call, causing endless meltdowns
    adapter = MockAdapter(responses=[
        # Attempt 1:
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]), # meltdown 1 triggers!
        
        # Attempt 2 (resumed):
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]), # meltdown 2 triggers!
        
        # Attempt 3 (resumed again):
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("loop_tool", {})]), # meltdown 3 triggers -> HARD_FAILED!
    ])

    runtime = SotisRuntime(
        adapter=adapter,
        tools=tools,
        tool_definitions=tool_defs,
        task_goal="Task",
        domain=Domain.SOFTWARE_ENGINEERING,
        loop_config=loop_config
    )

    result = runtime.run()

    assert result.success is False
    assert result.hard_failed is True
    assert result.total_resets == 2  # Max resets is capped at 2
    assert result.state.status == SessionStatus.HARD_FAILED


def test_checkpoint_incremental_baselines_reset(tmp_path):
    """Test that CheckpointManager baseline resets work incrementally through multiple resets."""
    tracked_file = tmp_path / "work.txt"
    tracked_file.write_text("initial\n", encoding="utf-8")

    tool_defs = [
        ToolDefinition("write_file", "Write file", {"path": {"type": "string"}, "content": {"type": "string"}})
    ]
    tools = {"write_file": mock_write_file}

    # Loop configuration to trigger meltdown on 2 repeats
    loop_config = LoopConfig(window_size=4, repeat_threshold=2)

    # Attempt 1:
    # 1. Modify work.txt to "initial\nedit_1\n"
    # 2. Call tool twice to trigger meltdown.
    # Attempt 2:
    # 3. Sotis resumes. Sotis baseline should now be "initial\nedit_1\n".
    # 4. Modify work.txt to "initial\nedit_1\nedit_2\n".
    # 5. Call tool twice to trigger meltdown 2.
    # 6. Checkpoint 2's diff should ONLY show "+edit_2" (incremental), not "+edit_1" AND "+edit_2"!
    adapter = MockAdapter(responses=[
        # Attempt 1:
        LLMResponse(text="", tool_calls=[ToolCall("write_file", {"path": str(tracked_file), "content": "initial\nedit_1\n"})]),
        LLMResponse(text="", tool_calls=[ToolCall("write_file", {"path": str(tracked_file), "content": "initial\nedit_1\n"})]), # meltdown 1!
        
        # Attempt 2:
        LLMResponse(text="", tool_calls=[ToolCall("write_file", {"path": str(tracked_file), "content": "initial\nedit_1\nedit_2\n"})]),
        LLMResponse(text="", tool_calls=[ToolCall("write_file", {"path": str(tracked_file), "content": "initial\nedit_1\nedit_2\n"})]), # meltdown 2!
        
        # Finish:
        LLMResponse(text="Finished", tool_calls=[])
    ])

    runtime = SotisRuntime(
        adapter=adapter,
        tools=tools,
        tool_definitions=tool_defs,
        task_goal="Write incrementally",
        workspace_paths=[str(tracked_file)],
        domain=Domain.SOFTWARE_ENGINEERING,
        loop_config=loop_config
    )

    result = runtime.run()

    assert result.success is True
    assert result.total_resets == 2
    
    # Inspect checkpoints in CheckpointManager
    mgr = runtime.checkpoint_manager
    assert mgr.checkpoint_count == 2
    
    cp1 = mgr.get_checkpoint(0)
    cp2 = mgr.get_checkpoint(1)

    diff1 = cp1.file_diffs[str(tracked_file.resolve())]
    diff2 = cp2.file_diffs[str(tracked_file.resolve())]

    # Checkpoint 1 should show edit_1 added
    assert "+edit_1" in diff1.unified_diff
    assert "+edit_2" not in diff1.unified_diff
    assert diff1.lines_added == 1

    # Checkpoint 2 should ONLY show edit_2 added relative to the post-Attempt-1 state!
    assert "+edit_2" in diff2.unified_diff
    assert "+edit_1" not in diff2.unified_diff  # Important: Incremental diff check!
    assert diff2.lines_added == 1
