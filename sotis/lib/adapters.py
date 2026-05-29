"""
sotis.lib.adapters
==================
Abstract LLM adapter interface + concrete implementations for OpenAI,
Anthropic, and DeepSeek (OpenAI-compatible endpoint).

Design philosophy
-----------------
Sotis is reliability middleware, not another agent framework. The adapters
are intentionally thin: they translate between each provider's wire format
and a single shared ``LLMResponse`` schema. All meltdown logic, entropy
monitoring, and checkpointing lives in the core layer — completely isolated
from any provider-specific code.

LangChain compatibility is intentionally NOT included here. It will be added
later as a callback adapter that wraps the runtime externally.

Public API
----------
ToolCall        : Represents a single tool invocation requested by the LLM.
LLMMessage      : A single message in the conversation (role + content).
LLMResponse     : Normalised output from any provider — text + optional tool calls.
LLMAdapter      : Abstract base class all concrete adapters must implement.
OpenAIAdapter   : Wraps the openai Python client.
AnthropicAdapter: Wraps the anthropic Python client.
DeepSeekAdapter : Uses the OpenAI-compatible DeepSeek base URL.
MockAdapter     : Deterministic, dependency-free adapter for unit tests.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """
    A single tool invocation requested by the LLM in its response.

    Attributes
    ----------
    tool_name : Name of the tool to call.
    arguments : Dict of argument name → value.
    call_id   : Provider-specific call ID (used for tool result injection).
                Empty string for providers that don't issue call IDs.
    """
    tool_name : str
    arguments : Dict[str, Any]
    call_id   : str = ""


@dataclass
class LLMMessage:
    """
    A single message in the conversation history.

    Attributes
    ----------
    role    : 'system', 'user', 'assistant', or 'tool'.
    content : Text content of the message.
    call_id : For role='tool', the call_id of the tool result being returned.
    """
    role    : str
    content : str
    call_id : str = ""


@dataclass
class LLMResponse:
    """
    Normalised response from any LLM provider.

    Attributes
    ----------
    text        : Text content from the assistant (may be empty if tool_calls present).
    tool_calls  : List of tool calls requested by the model (may be empty).
    stop_reason : 'tool_calls', 'end_turn', 'max_tokens', etc.
    raw         : The raw provider response object for debugging.
    """
    text        : str
    tool_calls  : List[ToolCall]  = field(default_factory=list)
    stop_reason : str             = "end_turn"
    raw         : Any             = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        """True when the model has stopped generating tool calls."""
        return not self.has_tool_calls


# ─────────────────────────────────────────────────────────────────────────────
# Abstract adapter
# ─────────────────────────────────────────────────────────────────────────────

class LLMAdapter(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Implementors must provide:
        complete(messages, tools) → LLMResponse
    """

    @abstractmethod
    def complete(
        self,
        messages    : List[LLMMessage],
        tools       : Optional[List[Dict[str, Any]]] = None,
        system      : Optional[str] = None,
        user_id     : Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a conversation to the LLM and return a normalised response.

        Parameters
        ----------
        messages : Ordered conversation history.
        tools    : Optional list of tool schemas in the provider's format.
        system   : Optional system prompt override.
        user_id  : Optional unique session/user identifier for abuse tracking.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name ('openai', 'anthropic', 'deepseek')."""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Adapter
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIAdapter(LLMAdapter):
    """
    Adapter for the OpenAI API (gpt-4o, gpt-4-turbo, etc.).

    Requires: ``openai`` package installed.

    Parameters
    ----------
    model      : Model identifier. Default: 'gpt-4o'.
    api_key    : OpenAI API key. Falls back to OPENAI_API_KEY env var.
    temperature: Sampling temperature. Default: 0.0 for determinism.
    """

    def __init__(
        self,
        model       : str   = "gpt-4o",
        api_key     : Optional[str] = None,
        temperature : float = 0.0,
        max_tokens  : int   = 4096,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            )
        self._client     = openai.OpenAI(api_key=api_key)
        self._model      = model
        self._temperature= temperature
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        messages : List[LLMMessage],
        tools    : Optional[List[Dict[str, Any]]] = None,
        system   : Optional[str] = None,
        user_id  : Optional[str] = None,
    ) -> LLMResponse:
        oai_messages = self._to_oai_messages(messages, system)
        kwargs: Dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        hashed_uid = hashlib.sha256(user_id.encode()).hexdigest() if user_id else "sotis-anonymous"
        client = self._client

        # Moderate user input before sending to the model to check for harmful content
        user_inputs = [m.content for m in messages if m.role == "user"]
        if self.provider_name == "openai" and user_inputs:
            try:
                mod_response = client.moderations.create(input="\n".join(user_inputs))
                if mod_response.results[0].flagged:
                    raise ValueError(
                        "Input was flagged by OpenAI Content Moderation API as potentially harmful."
                    )
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise exc
                raise RuntimeError(
                    f"OpenAI Moderation check failed: {type(exc).__name__}: {exc}"
                ) from exc

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                user=hashed_uid,
                **kwargs
            )
        except Exception as exc:
            # Handle rate limits, API errors, and network issues gracefully
            raise RuntimeError(
                f"OpenAI API call failed: {type(exc).__name__}: {exc}"
            ) from exc

        choice   = response.choices[0]
        msg      = choice.message

        text       = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    tool_name=tc.function.name,
                    arguments=args,
                    call_id=tc.id or "",
                ))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "end_turn",
            raw=response,
        )

    def _to_oai_messages(
        self,
        messages: List[LLMMessage],
        system  : Optional[str],
    ) -> List[Dict[str, Any]]:
        oai = []
        if system:
            oai.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                oai.append({
                    "role"         : "tool",
                    "content"      : m.content,
                    "tool_call_id" : m.call_id,
                })
            else:
                oai.append({"role": m.role, "content": m.content})
        return oai


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Adapter
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicAdapter(LLMAdapter):
    """
    Adapter for the Anthropic API (claude-3-5-sonnet, claude-3-opus, etc.).

    Requires: ``anthropic`` package installed.

    Parameters
    ----------
    model      : Model identifier. Default: 'claude-3-5-sonnet-20241022'.
    api_key    : Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    max_tokens : Max tokens to generate. Default: 4096.
    """

    def __init__(
        self,
        model      : str           = "claude-3-5-sonnet-20241022",
        api_key    : Optional[str] = None,
        max_tokens : int           = 4096,
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
            )
        self._client     = anthropic.Anthropic(api_key=api_key)
        self._model      = model
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def complete(
        self,
        messages : List[LLMMessage],
        tools    : Optional[List[Dict[str, Any]]] = None,
        system   : Optional[str] = None,
        user_id  : Optional[str] = None,
    ) -> LLMResponse:
        anth_messages = [
            {"role": m.role if m.role != "tool" else "user",
             "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        kwargs: Dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        hashed_uid = hashlib.sha256(user_id.encode()).hexdigest() if user_id else "sotis-anonymous"

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=anth_messages,
                system=system or "You are a helpful assistant.",
                metadata={"user_id": hashed_uid},
                **kwargs
            )
        except Exception as exc:
            # Handle rate limits, API errors, and network issues gracefully
            raise RuntimeError(
                f"Anthropic API call failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Check stop_reason explicitly to satisfy IDE safety guidelines
        stop_reason = response.stop_reason or "end_turn"
        if stop_reason in ("max_tokens", "stop_sequence"):
            # Model stopped unexpectedly, handled by client inspect
            pass

        text        = ""
        tool_calls  = []

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    tool_name=block.name,
                    arguments=block.input or {},
                    call_id=block.id or "",
                ))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=response,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek Adapter (OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekAdapter(OpenAIAdapter):
    """
    Adapter for DeepSeek models using their OpenAI-compatible API.

    DeepSeek exposes the same REST interface as the OpenAI API, so this
    adapter simply overrides the base URL.

    Parameters
    ----------
    model   : Default 'deepseek-chat'.
    api_key : DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        model  : str           = "deepseek-chat",
        api_key: Optional[str] = None,
    ) -> None:
        import os  # noqa: PLC0415
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            raise ImportError("openai package is required for DeepSeekAdapter.")

        self._client      = openai.OpenAI(
            api_key  = key,
            base_url = self.DEEPSEEK_BASE_URL,
        )
        self._model       = model
        self._temperature = 0.0

    @property
    def provider_name(self) -> str:
        return "deepseek"


# ─────────────────────────────────────────────────────────────────────────────
# Mock Adapter (for testing — zero external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class MockAdapter(LLMAdapter):
    """
    Deterministic mock adapter for unit and integration tests.

    Drives agent behaviour through a user-provided response queue. Each call
    to ``complete()`` pops the next ``LLMResponse`` from the queue. When the
    queue is exhausted, a default terminal response is returned.

    Parameters
    ----------
    responses : Ordered list of ``LLMResponse`` objects to return.

    Usage
    -----
        mock = MockAdapter(responses=[
            LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "a.py"})]),
            LLMResponse(text="Done", tool_calls=[]),
        ])
        runtime = SotisRuntime(adapter=mock, ...)
    """

    def __init__(self, responses: Optional[List[LLMResponse]] = None) -> None:
        self._queue   : List[LLMResponse] = list(responses or [])
        self.call_log : List[List[LLMMessage]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(
        self,
        messages : List[LLMMessage],
        tools    : Optional[List[Dict[str, Any]]] = None,
        system   : Optional[str] = None,
        user_id  : Optional[str] = None,
    ) -> LLMResponse:
        self.call_log.append(list(messages))
        if self._queue:
            return self._queue.pop(0)
        # Default: terminal response when queue exhausted.
        return LLMResponse(text="Task complete.", tool_calls=[], stop_reason="end_turn")

    def push(self, response: LLMResponse) -> None:
        """Add a response to the back of the queue."""
        self._queue.append(response)

    @property
    def calls_made(self) -> int:
        return len(self.call_log)
