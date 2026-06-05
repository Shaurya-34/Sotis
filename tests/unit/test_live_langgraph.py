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


# ─────────────────────────────────────────────────────────────────────────────
# New-feature telemetry: tokens, adaptive threshold, rollback target
# ─────────────────────────────────────────────────────────────────────────────

def test_step_event_tokens_roundtrip():
    """StepEvent.tokens must survive model_dump() (what log_step serializes)."""
    ev = StepEvent(step_index=0, tool_name="t", tokens=1234)
    dumped = ev.model_dump()
    assert dumped["tokens"] == 1234
    # None is preserved too
    ev2 = StepEvent(step_index=1, tool_name="t")
    assert ev2.model_dump()["tokens"] is None


def test_meltdown_signal_new_fields_default_and_dump():
    """The new observability fields exist, default safely, and serialize."""
    from sotis.core.schemas import MeltdownSignal, MeltdownReason
    sig = MeltdownSignal(
        session_id="s", subtask_id=None, triggered_at_step=3,
        reason=MeltdownReason.TOOL_LOOP,
    )
    d = sig.model_dump()
    assert d["effective_threshold"] is None
    assert d["adaptive_active"] is False
    assert d["rollback_target"] is None
    # and they carry values when set
    sig2 = MeltdownSignal(
        session_id="s", subtask_id=None, triggered_at_step=3,
        reason=MeltdownReason.HIGH_ENTROPY,
        effective_threshold=2.31, adaptive_active=True,
        rollback_target="verified-good #4",
    )
    d2 = sig2.model_dump()
    assert d2["effective_threshold"] == 2.31
    assert d2["adaptive_active"] is True
    assert d2["rollback_target"] == "verified-good #4"


def test_tokens_thread_from_usage_metadata():
    """A real AIMessage carrying usage_metadata should populate StepEvent.tokens."""
    guard = SotisLangGraphGuard(task_goal="token threading")
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "do_x", "args": {}, "id": "c-1"}],
        id="m-1",
    )
    # Real langchain AIMessage accepts usage_metadata; if the stub is in use,
    # set the attribute directly so the guard's getattr path still works.
    try:
        ai.usage_metadata = {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}
    except Exception:
        setattr(ai, "usage_metadata", {"total_tokens": 140})

    messages = [ai, ToolMessage(content="ok", name="do_x", tool_call_id="c-1", id="m-2")]
    guard.guard_node({"messages": messages})
    assert len(guard.state.trajectory) == 1
    assert guard.state.trajectory[0].tokens == 140


def test_meltdown_records_rollback_target(tmp_path):
    """On a loop meltdown, the logged signal carries a rollback_target string."""
    from sotis.core.checkpoint import python_imports_cleanly
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n")
    guard = SotisLangGraphGuard(
        task_goal="rollback target",
        workspace_paths=[str(f)],
        checkpoint_invariant=python_imports_cleanly,
    )
    # one clean step so a verified checkpoint can be committed
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c-0"}], id="m-0"),
        ToolMessage(content="x = 1", name="read_file", tool_call_id="c-0", id="t-0"),
    ]
    guard.guard_node({"messages": msgs})
    # force a loop
    update = {}
    for i in range(3):
        loop = msgs + [
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"c": "y"}, "id": f"l-{i}"}], id=f"ai-{i}"),
            ToolMessage(content="err", name="write_file", tool_call_id=f"l-{i}", id=f"tt-{i}"),
        ]
        update = guard.guard_node({"messages": loop})
    assert guard.total_resets == 1
    # the recorded meltdown must carry a rollback_target
    sig = guard.state.meltdown_log[-1]
    assert sig.rollback_target in ("baseline",) or sig.rollback_target.startswith("verified-good")
