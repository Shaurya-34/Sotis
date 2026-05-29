"""
sotis.lib.runtime
=================
Custom ReAct-style execution runtime with integrated Sotis meltdown interception.

Architecture
------------
``SotisRuntime`` is the developer-facing entry point. It wraps a user-provided
``LLMAdapter`` and a dict of tool functions, then runs a standard
Observe → Think → Act loop. Every tool call is intercepted before and after
execution to:
    1. Create a ``StepEvent`` and push it to both monitors.
    2. Check for meltdown signals (entropy OR loop).
    3. On meltdown: snapshot workspace → distil context → inject reset prompt
       → continue execution from the new distilled state.
    4. Enforce the 2-reset hard cap per subtask; hard-fail beyond that.

MOP = Meltdown-Observable-Protocol
    A MOP-triggered reset is fully transparent: the agent in the next
    context window sees a clean, structured briefing rather than the raw
    chaotic history that caused the spiral.

Public API
----------
ToolDefinition  : Schema for a callable tool (name, description, parameters).
RunResult       : Final outcome of a SotisRuntime.run() call.
SotisRuntime    : The primary runtime class.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sotis.core.checkpoint import CheckpointManager
from sotis.core.entropy import EntropyConfig, SessionEntropyTracker
from sotis.core.loops import LoopConfig, SessionLoopTracker
from sotis.core.reset import ContextResetter, DistillationConfig, DistillationResult
from sotis.core.schemas import (
    Domain,
    ExecutionState,
    MeltdownReason,
    MeltdownSignal,
    SessionStatus,
    StepEvent,
    Subtask,
)
from sotis.lib.adapters import LLMAdapter, LLMMessage, LLMResponse, ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# Supporting types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """
    Schema for a tool that the LLM may call.

    Attributes
    ----------
    name        : Unique tool name (matches function key in tool_registry).
    description : Human-readable description shown to the LLM.
    parameters  : JSON-Schema-compatible dict describing the tool arguments.
    """
    name       : str
    description: str
    parameters : Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name"       : self.name,
                "description": self.description,
                "parameters" : self.parameters,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name"       : self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class RunResult:
    """
    Final outcome of a ``SotisRuntime.run()`` call.

    Attributes
    ----------
    session_id          : Unique session identifier.
    success             : True if the agent reached a terminal state normally.
    hard_failed         : True if the 2-reset cap was exhausted on a subtask.
    total_steps         : Total tool calls made across all execution contexts.
    total_resets        : Number of context resets performed.
    final_response      : The last text response from the LLM.
    distillation_results: List of DistillationResult objects from each reset.
    duration_ms         : Wall-clock duration in milliseconds.
    state               : The final ExecutionState for inspection/logging.
    """
    session_id           : str
    success              : bool
    hard_failed          : bool
    total_steps          : int
    total_resets         : int
    final_response       : str
    distillation_results : List[DistillationResult]
    duration_ms          : float
    state                : ExecutionState


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 150    # Global step ceiling — prevents infinite loops.


class SotisRuntime:
    """
    MOP-Triggered ReAct execution runtime.

    Usage
    -----
        runtime = SotisRuntime(
            adapter=OpenAIAdapter(model="gpt-4o"),
            tools={"read_file": read_file_fn, "write_code": write_code_fn},
            tool_definitions=[ToolDefinition("read_file", ...)],
            task_goal="Refactor the legacy auth module to use async/await.",
            workspace_paths=["src/auth.py", "tests/test_auth.py"],
        )
        result = runtime.run()
        print(result.success, result.total_resets)

    Parameters
    ----------
    adapter             : LLMAdapter instance (OpenAI / Anthropic / DeepSeek / Mock).
    tools               : Dict mapping tool_name → Python callable.
    tool_definitions    : List of ToolDefinition objects (schema for the LLM).
    task_goal           : High-level task description (injected into system prompt).
    workspace_paths     : Paths to track for incremental checkpointing.
    domain              : Task domain (SE / WR / DP).
    entropy_config      : Custom EntropyConfig (defaults to N=5, H=1.5).
    loop_config         : Custom LoopConfig (defaults to window=6, threshold=3).
    distillation_config : Custom DistillationConfig.
    max_steps           : Hard step ceiling. Default: 150.
    session_id          : Custom session ID. Auto-generated if not provided.
    """

    def __init__(
        self,
        adapter              : LLMAdapter,
        tools                : Dict[str, Callable[..., Any]],
        tool_definitions     : List[ToolDefinition],
        task_goal            : str,
        workspace_paths      : Optional[List[str]] = None,
        domain               : Domain = Domain.UNKNOWN,
        entropy_config       : Optional[EntropyConfig] = None,
        loop_config          : Optional[LoopConfig] = None,
        distillation_config  : Optional[DistillationConfig] = None,
        max_steps            : int = _DEFAULT_MAX_STEPS,
        session_id           : Optional[str] = None,
    ) -> None:
        self._adapter              = adapter
        self._tools                = tools
        self._tool_definitions     = tool_definitions
        self._task_goal            = task_goal
        self._workspace_paths      = workspace_paths or []
        self._max_steps            = max_steps

        # Session identity.
        self.session_id = session_id or f"sotis-{uuid.uuid4().hex[:12]}"

        # Execution state.
        self._state = ExecutionState(
            session_id=self.session_id,
            domain=domain,
        )

        # Monitors (stateful wrappers).
        self._entropy_tracker = SessionEntropyTracker(entropy_config)
        self._loop_tracker    = SessionLoopTracker(loop_config)

        # Phase 2 components.
        self._checkpoint_mgr  = CheckpointManager()
        self._resetter        = ContextResetter(distillation_config)

        # Track resets per subtask.
        self._active_subtask_resets : int = 0

        # Distillation audit trail.
        self._distillation_results : List[DistillationResult] = []

        # Initialise file tracking.
        if self._workspace_paths:
            self._checkpoint_mgr.track(self._workspace_paths)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> RunResult:
        """
        Execute the ReAct loop until terminal state, meltdown cap, or step limit.

        Returns
        -------
        RunResult
        """
        start_ms = time.time() * 1000

        # Build initial conversation.
        messages  = self._build_initial_messages()
        tool_schemas = self._get_tool_schemas()

        final_response = ""
        hard_failed    = False
        step_index     = 0

        while step_index < self._max_steps:
            # ── LLM call ─────────────────────────────────────────────────────
            llm_resp = self._adapter.complete(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                system=self._build_system_prompt(),
                user_id=self.session_id,
            )

            if llm_resp.is_final:
                final_response = llm_resp.text
                self._state.status = SessionStatus.COMPLETED
                break

            # ── Process each tool call ────────────────────────────────────────
            tool_results: List[LLMMessage] = []
            meltdown_detected = False

            for tool_call in llm_resp.tool_calls:
                # Execute the tool.
                result_text = self._execute_tool(tool_call)

                # Build and record the StepEvent.
                event = StepEvent(
                    step_index    =step_index,
                    tool_name     =tool_call.tool_name,
                    tool_args     =tool_call.arguments,
                    result_summary=result_text[:400] if result_text else None,
                    subtask_id    =self._state.active_subtask_id,
                )
                self._state.record_step(event)

                # Push through monitors.
                entropy_result = self._entropy_tracker.push_event(event)
                loop_result    = self._loop_tracker.push_event(event)

                # ── Meltdown check ────────────────────────────────────────────
                if entropy_result.meltdown_detected or loop_result.meltdown_detected:
                    meltdown_detected = True

                    # Determine reason.
                    if entropy_result.meltdown_detected and loop_result.meltdown_detected:
                        reason = MeltdownReason.COMBINED
                    elif entropy_result.meltdown_detected:
                        reason = entropy_result.reason or MeltdownReason.HIGH_ENTROPY
                    else:
                        reason = MeltdownReason.TOOL_LOOP

                    self._active_subtask_resets += 1
                    reset_attempt = self._active_subtask_resets

                    # Hard cap check.
                    if reset_attempt > 2:
                        self._state.status = SessionStatus.HARD_FAILED
                        hard_failed = True
                        break

                    # Build meltdown signal.
                    signal = MeltdownSignal(
                        session_id       =self.session_id,
                        subtask_id       =self._state.active_subtask_id,
                        triggered_at_step=step_index,
                        reason           =reason,
                        entropy_value    =entropy_result.entropy,
                        loop_tool        =loop_result.dominant_tool_name,
                        loop_count       =loop_result.dominant_count,
                        reset_attempt    =reset_attempt,
                    )
                    self._state.record_meltdown(signal)

                    # Snapshot & distil.
                    checkpoint = self._checkpoint_mgr.snapshot(self._state, signal)
                    distillation = self._resetter.distill(
                        state=self._state,
                        checkpoint=checkpoint,
                        task_goal=self._task_goal,
                    )
                    self._distillation_results.append(distillation)

                    # Reset monitors and baselines.
                    self._entropy_tracker.reset()
                    self._loop_tracker.reset()
                    self._checkpoint_mgr.reset_baselines()

                    # Inject distilled prompt as new conversation start.
                    messages = self._build_reset_messages(distillation.prompt)
                    self._state.status = SessionStatus.RESUMED
                    break   # Restart the LLM loop with new context.

                # Append tool result to messages.
                tool_results.append(LLMMessage(
                    role   ="tool",
                    content=result_text,
                    call_id=tool_call.call_id,
                ))
                step_index += 1

            if hard_failed:
                break

            if meltdown_detected:
                # Reset already injected — continue the outer while loop.
                step_index += 1
                continue

            # Append assistant + tool results to conversation.
            if llm_resp.text or llm_resp.tool_calls:
                messages.append(LLMMessage(role="assistant", content=llm_resp.text))
            messages.extend(tool_results)

        end_ms = time.time() * 1000

        return RunResult(
            session_id=self.session_id,
            success=self._state.status == SessionStatus.COMPLETED,
            hard_failed=hard_failed,
            total_steps=self._state.step_count,
            total_resets=self._state.total_resets,
            final_response=final_response,
            distillation_results=self._distillation_results,
            duration_ms=end_ms - start_ms,
            state=self._state,
        )

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return its result as a string."""
        fn = self._tools.get(tool_call.tool_name)
        if fn is None:
            return f"[ERROR] Tool '{tool_call.tool_name}' not found in registry."
        try:
            result = fn(**tool_call.arguments)
            return str(result) if result is not None else ""
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] Tool '{tool_call.tool_name}' raised: {type(exc).__name__}: {exc}"

    # ── Message builders ──────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return (
            "You are a reliable AI assistant. Complete the given task by calling "
            "the available tools. Be methodical and focused. When you have "
            "finished, respond with your final answer in plain text without any "
            "tool calls."
        )

    def _build_initial_messages(self) -> List[LLMMessage]:
        return [
            LLMMessage(
                role   ="user",
                content=f"Please complete the following task:\n\n{self._task_goal}",
            )
        ]

    def _build_reset_messages(self, distilled_prompt: str) -> List[LLMMessage]:
        """Build a fresh conversation history from the distilled prompt."""
        return [
            LLMMessage(
                role   ="user",
                content=distilled_prompt,
            )
        ]

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas in the format appropriate for the adapter."""
        provider = self._adapter.provider_name
        if provider == "anthropic":
            return [t.to_anthropic_schema() for t in self._tool_definitions]
        # OpenAI, DeepSeek, Mock — all use OpenAI format.
        return [t.to_openai_schema() for t in self._tool_definitions]

    # ── State accessors for external inspection ───────────────────────────────

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_mgr

    @property
    def distillation_results(self) -> List[DistillationResult]:
        return list(self._distillation_results)
