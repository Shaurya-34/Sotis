"""
sotis.lib.langgraph_integration
==============================
Native Sotis middleware node and callbacks for LangGraph.

This module provides the `SotisLangGraphGuard` class, which integrates Sotis'
MOP-triggered resilience observability into a LangGraph ReAct workflow.

Architecture
------------
The `sotis_guard` node intercepts the graph execution. By reviewing the state
messages, it matches `ToolMessage` results with preceding `AIMessage` tool calls,
translates them to `StepEvent` records, and pushes them to Sotis monitors.

If a loop or high entropy is detected:
    1. A workspace snapshot is captured using ``CheckpointManager``.
    2. Any files modified during the failed cycle are rolled back to the last stable state.
    3. The context is distilled into a compact Markdown prompt.
    4. Sotis prunes the message history using `RemoveMessage` signals and starts the LLM
       fresh with the distilled resumption briefing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Union

# Try to import LangChain/LangGraph messages cleanly
try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        RemoveMessage,
        SystemMessage,
        ToolMessage,
    )
except ImportError:
    # Stubs for CI/testing when packages are not fully loaded
    class BaseMessage:
        def __init__(self, content: str, id: Optional[str] = None):
            self.content = content
            self.id = id or str(uuid.uuid4())

    class AIMessage(BaseMessage):
        def __init__(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None, id: Optional[str] = None):
            super().__init__(content, id)
            self.tool_calls = tool_calls or []

    class ToolMessage(BaseMessage):
        def __init__(self, content: str, name: str, tool_call_id: str, id: Optional[str] = None):
            super().__init__(content, id)
            self.name = name
            self.tool_call_id = tool_call_id

    class HumanMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class RemoveMessage(BaseMessage):
        def __init__(self, id: Optional[str] = None, content: str = "", **kwargs):
            super().__init__(content=content, id=id)


from sotis.core.checkpoint import CheckpointManager
from sotis.core.entropy import EntropyConfig, SessionEntropyTracker
from sotis.core.loops import LoopConfig, SessionLoopTracker, WorkspaceDensityGuard
from sotis.core.reset import ContextResetter, DistillationConfig
from sotis.core.schemas import (
    Domain,
    ExecutionState,
    MeltdownReason,
    MeltdownSignal,
    SessionStatus,
    StepEvent,
)
from sotis.obs.logger import SessionLogger

logger = logging.getLogger("sotis.langgraph")


class SotisLangGraphGuard:
    """
    State node guard for LangGraph.

    This class maintains Sotis' monitoring metrics (entropy, loops, checkpoints)
    and acts as a gatekeeper node in the graph, checking for meltdowns and modifying
    the conversation thread upon context reset.

    Parameters
    ----------
    task_goal           : High-level goal of the agent task.
    workspace_paths     : Files to track for incremental workspace checkpointing.
    domain              : Task domain (Domain.SOFTWARE_ENGINEERING, Domain.WEB_RESEARCH, etc.).
    entropy_config      : Custom EntropyConfig overrides.
    loop_config         : Custom LoopConfig overrides.
    distillation_config : Custom DistillationConfig overrides.
    session_id          : Optional custom session ID.
    """

    def __init__(
        self,
        task_goal: str,
        workspace_paths: Optional[List[str]] = None,
        domain: Domain = Domain.UNKNOWN,
        entropy_config: Optional[EntropyConfig] = None,
        loop_config: Optional[LoopConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.task_goal = task_goal
        self.workspace_paths = workspace_paths or []
        self.session_id = session_id or f"sotis-lg-{uuid.uuid4().hex[:8]}"

        # Initialize core engine modules
        self.state = ExecutionState(session_id=self.session_id, domain=domain)
        self.entropy_tracker = SessionEntropyTracker(entropy_config)
        self.loop_tracker = SessionLoopTracker(loop_config)
        self.density_guard = WorkspaceDensityGuard(max_consecutive_edits=3)
        self.checkpoint_mgr = CheckpointManager()
        self.resetter = ContextResetter(distillation_config)
        self.telemetry = SessionLogger(self.session_id)


        # Resets tracker
        self.total_resets = 0

        # Process step counters
        self._processed_message_ids: Set[str] = set()

        if self.workspace_paths:
            self.checkpoint_mgr.track(self.workspace_paths)

    def guard_node(self, state: Any) -> Dict[str, Any]:
        """
        Intercepts state messages, updates trackers, and triggers a context reset if necessary.

        This node should be wired directly after the tool node execution node in LangGraph.

        Parameters
        ----------
        state : The LangGraph current state dictionary. Must contain a 'messages' key
                which is a List of BaseMessage objects.

        Returns
        -------
        A State update dictionary. Under normal conditions, returns an empty update `{}`.
        During meltdown resets, returns a dictionary containing:
            1. RemoveMessage updates to prune old raw tokens.
            2. A single new HumanMessage containing Sotis' distilled prompt.
        """
        messages: List[BaseMessage] = state.get("messages", [])
        if not messages:
            return {}

        # 1. Detect any new completed tool executions
        new_events = self._parse_new_tool_events(messages)

        meltdown_detected = False
        meltdown_signal: Optional[MeltdownSignal] = None

        # Check for text-only reasoning loops / pseudo-tool loops
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if len(ai_messages) >= 3:
            last_3_texts = []
            for m in ai_messages[-3:]:
                content = m.content if isinstance(m.content, str) else ""
                normalized = " ".join(content.split()).strip()
                if normalized:
                    last_3_texts.append(normalized)

            if len(last_3_texts) == 3:
                is_repetition = (last_3_texts[-1] == last_3_texts[-2]) or (last_3_texts[-1] == last_3_texts[-3])
                has_pseudo_tool = '[{"name":' in last_3_texts[-1] or '{"name":' in last_3_texts[-1]

                if is_repetition or (has_pseudo_tool and len(ai_messages) >= 4):
                    meltdown_detected = True
                    self.total_resets += 1

                    if self.total_resets > 2:
                        self.state.status = SessionStatus.HARD_FAILED
                        logger.error(f"[Sotis] Meltdown cap exhausted (resets: {self.total_resets}). Hard failing.")
                        raise RuntimeError("Sotis intercepted a terminal agent meltdown: Context reset limit (2) exceeded.")

                    meltdown_signal = MeltdownSignal(
                        session_id=self.session_id,
                        subtask_id=None,
                        triggered_at_step=len(self.state.trajectory),
                        reason=MeltdownReason.TOOL_LOOP,
                        entropy_value=2.0,
                        loop_tool="text_reasoning_loop",
                        loop_count=3,
                        reset_attempt=self.total_resets,
                    )
                    self.state.record_meltdown(meltdown_signal)
                    self.telemetry.log_meltdown(meltdown_signal)
                    self.telemetry.log_state(self.state)

        # Process new tool events if no text loop meltdown has been triggered yet
        if not meltdown_detected and new_events:
            for event in new_events:
                self.state.record_step(event)
                self.telemetry.log_step(event)

                # Evaluate metrics
                entropy_res = self.entropy_tracker.push_event(event)
                loop_res = self.loop_tracker.push_event(event)
                density_triggered = self.density_guard.push_event(event)

                if entropy_res.meltdown_detected or loop_res.meltdown_detected or density_triggered:
                    meltdown_detected = True

                    # Determine reason
                    if entropy_res.meltdown_detected and loop_res.meltdown_detected:
                        reason = MeltdownReason.COMBINED
                    elif entropy_res.meltdown_detected:
                        reason = entropy_res.reason or MeltdownReason.HIGH_ENTROPY
                    else:
                        reason = MeltdownReason.TOOL_LOOP

                    self.total_resets += 1

                    # Cap enforcement
                    if self.total_resets > 2:
                        self.state.status = SessionStatus.HARD_FAILED
                        logger.error(f"[Sotis] Meltdown cap exhausted (resets: {self.total_resets}). Hard failing.")
                        raise RuntimeError("Sotis intercepted a terminal agent meltdown: Context reset limit (2) exceeded.")

                    # Capture meltdown signal
                    meltdown_signal = MeltdownSignal(
                        session_id=self.session_id,
                        subtask_id=event.subtask_id,
                        triggered_at_step=event.step_index,
                        reason=reason,
                        entropy_value=entropy_res.entropy,
                        loop_tool=self.density_guard.triggered_file or loop_res.dominant_tool_name,
                        loop_count=3 if density_triggered else loop_res.dominant_count,
                        reset_attempt=self.total_resets,
                    )
                    self.state.record_meltdown(meltdown_signal)
                    self.telemetry.log_meltdown(meltdown_signal)
                    self.telemetry.log_state(self.state)
                    break

        # 2. If a meltdown was triggered, perform checkpoint snapshot, file rollback, and context reset
        if meltdown_detected and meltdown_signal:
            logger.warning(f"[Sotis] Meltdown intercepted! Reason: {meltdown_signal.reason}. Triggering context reset.")

            # Snapshot current files and revert uncommitted edits back to baseline
            checkpoint = self.checkpoint_mgr.snapshot(self.state, meltdown_signal)
            
            # Rollback modified files to prevent agent from inheriting a broken environment
            if checkpoint.modified_files:
                logger.warning(f"[Sotis] Rolling back modified files to last stable state: {list(checkpoint.modified_files)}")
                self.checkpoint_mgr.rollback()

            # Distill the prompt
            distillation = self.resetter.distill(
                state=self.state,
                checkpoint=checkpoint,
                task_goal=self.task_goal,
            )

            # Reset monitors for the clean slate run
            self.entropy_tracker.reset()
            self.loop_tracker.reset()
            self.density_guard.reset()
            self.checkpoint_mgr.reset_baselines()

            # Transition from MELTDOWN → RESUMED
            self.state.status = SessionStatus.RESUMED

            # 3. Build LangGraph state message modifications (pruning message history)
            return self._build_reset_state_update(messages, distillation.prompt)

        self.telemetry.log_state(self.state)
        return {}


    def _parse_new_tool_events(self, messages: List[BaseMessage]) -> List[StepEvent]:
        """Match ToolMessages with preceding AIMessages to build StepEvents."""
        events: List[StepEvent] = []

        # Construct a map of tool_call_id -> AIMessage details
        ai_tool_calls: Dict[str, Dict[str, Any]] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    call_id = tc.get("id") or tc.get("call_id")
                    if isinstance(call_id, str):
                        ai_tool_calls[call_id] = {
                            "name": tc.get("name"),
                            "args": tc.get("args") or tc.get("arguments") or {},
                        }

        # Scan for unprocessed ToolMessages
        for msg in messages:
            if isinstance(msg, ToolMessage):
                msg_id = getattr(msg, "id", None) or f"tool-{msg.tool_call_id}"
                if msg_id in self._processed_message_ids:
                    continue

                call_id = msg.tool_call_id
                ai_info = ai_tool_calls.get(call_id, {"name": msg.name, "args": {}})

                tool_name = str(ai_info.get("name") or msg.name or "unknown")
                event = StepEvent(
                    step_index=len(self.state.trajectory),
                    tool_name=tool_name,
                    tool_args=ai_info["args"],
                    result_summary=str(msg.content)[:400] if msg.content else None,
                )
                events.append(event)
                self._processed_message_ids.add(msg_id)

        return events

    def _build_reset_state_update(self, messages: List[BaseMessage], distilled_prompt: str) -> Dict[str, Any]:
        """Prune conversation history and insert Sotis resumption prompt."""
        # 1. Emit RemoveMessage for all messages in graph state to completely reduce token overhead
        remove_updates: List[RemoveMessage] = []
        for msg in messages:
            msg_id = getattr(msg, "id", None)
            if msg_id:
                remove_updates.append(RemoveMessage(id=msg_id))

        # 2. Append standard HumanMessage with the self-contained distilled prompt
        new_human_message = HumanMessage(
            content=distilled_prompt,
            id=f"sotis-resume-{self.total_resets}",
        )

        logger.info(f"[Sotis] Emitted {len(remove_updates)} RemoveMessage updates and context reset prompt.")
        return {
            "messages": remove_updates + [new_human_message],
        }
