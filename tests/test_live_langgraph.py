"""
tests/test_live_langgraph
=========================
Offline automated unit tests for Sotis' LangGraph integration (Track 1).
"""

from __future__ import annotations

import os
from typing import Dict, List

import pytest

from sotis.core.schemas import Domain, SessionStatus, StepEvent
from sotis.lib.langgraph_integration import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SotisLangGraphGuard,
    ToolMessage,
)


def test_sotis_guard_initialization():
    """Verify default properties and setup of the Sotis LangGraph node guard."""
    guard = SotisLangGraphGuard(
        task_goal="Fix calculation bugs",
        workspace_paths=[],
        domain=Domain.SOFTWARE_ENGINEERING,
    )
    assert guard.task_goal == "Fix calculation bugs"
    assert guard.total_resets == 0
    assert guard.state.status == SessionStatus.RUNNING
    assert "sotis-lg-" in guard.session_id


def test_sotis_guard_normal_step():
    """Verify normal tool executions are parsed and monitored without triggering meltdown."""
    guard = SotisLangGraphGuard(task_goal="Add multiply operation")

    # Construct messages representing a standard tool run
    messages: List[BaseMessage] = [
        HumanMessage(content="Calculate 2 * 3", id="msg-1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}, "id": "call-1"}],
            id="msg-2",
        ),
        ToolMessage(content="6", name="multiply", tool_call_id="call-1", id="msg-3"),
    ]

    state = {"messages": messages}
    update = guard.guard_node(state)

    # Under normal operation, the guard should not mutate state
    assert update == {}
    assert len(guard.state.trajectory) == 1

    event: StepEvent = guard.state.trajectory[0]
    assert event.tool_name == "multiply"
    assert event.tool_args == {"a": 2, "b": 3}
    assert event.result_summary == "6"


def test_sotis_guard_meltdown_and_reset(tmp_path):
    """Verify that an execution loop triggers meltdown, files are reverted, and messages are pruned."""
    # Setup a tracked temporary test file
    test_file = tmp_path / "code.py"
    test_file.write_text("initial_state = True\n")

    guard = SotisLangGraphGuard(
        task_goal="Fix standard cycle",
        workspace_paths=[str(test_file)],
        domain=Domain.SOFTWARE_ENGINEERING,
    )

    # 1. Simulate stable first step
    messages = [
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "code.py"}, "id": "c-1"}], id="m-1"),
        ToolMessage(content="initial_state = True\n", name="read_file", tool_call_id="c-1", id="m-2"),
    ]
    guard.guard_node({"messages": messages})
    assert len(guard.state.trajectory) == 1

    # Apply some uncommitted file changes (unstable edit)
    test_file.write_text("initial_state = True\n# broken modification\n")

    # 2. Simulate 3 exact identical tool calls to trigger the loop detector
    for i in range(3):
        loop_messages = messages + [
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"content": "fix"}, "id": f"lc-{i}"}], id=f"m-ai-{i}"),
            ToolMessage(content="error", name="write_file", tool_call_id=f"lc-{i}", id=f"m-t-{i}"),
        ]
        # The third push will trigger meltdown
        update = guard.guard_node({"messages": loop_messages})

    # Meltdown asserts
    assert guard.total_resets == 1
    assert guard.state.status == SessionStatus.RESUMED
    assert "messages" in update

    # The updates list must contain RemoveMessage objects for old messages and a single resumption HumanMessage
    remove_msgs = [m for m in update["messages"] if isinstance(m, RemoveMessage)]
    human_msgs = [m for m in update["messages"] if isinstance(m, HumanMessage)]

    # Pruned history verification
    assert len(remove_msgs) > 0
    assert len(human_msgs) == 1

    # Resumption briefing checks
    distilled_prompt = human_msgs[0].content
    assert "Context Reset Notice (Reset #1/2)" in distilled_prompt
    assert "repeated identical tool calls" in distilled_prompt
    assert "Fix standard cycle" in distilled_prompt
    assert "Instructions" in distilled_prompt

    # Rollback assertion: File must have reverted back to its stable tracked base!
    assert test_file.read_text() == "initial_state = True\n"
