# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Core types for the LLM infrastructure.

Defines message formats, responses, and streaming types used across all providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format for API calls."""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        """Create from API response dict."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", data.get("input_tokens", 0)),
            completion_tokens=data.get(
                "completion_tokens", data.get("output_tokens", 0)
            ),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[str] = None
    provider: str = ""
    latency_ms: float = 0.0
    raw: Optional[Any] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens or (
            self.usage.prompt_tokens + self.usage.completion_tokens
        )


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str
    delta: str
    finish_reason: Optional[str] = None
    is_final: bool = False
    usage: Optional[Usage] = None


class StreamingResponse:
    """Iterator wrapper for streaming LLM responses."""

    def __init__(
        self,
        iterator: Iterator[StreamChunk],
        model: str = "",
        provider: str = "",
    ):
        self._iterator = iterator
        self._model = model
        self._provider = provider
        self._chunks: List[StreamChunk] = []
        self._content = ""
        self._finished = False

    def __iter__(self) -> Iterator[StreamChunk]:
        for chunk in self._iterator:
            self._chunks.append(chunk)
            self._content += chunk.delta
            if chunk.is_final:
                self._finished = True
            yield chunk

    @property
    def content(self) -> str:
        """Get accumulated content (only complete after iteration)."""
        return self._content

    @property
    def is_finished(self) -> bool:
        return self._finished

    def to_response(self) -> LLMResponse:
        """Convert to a complete LLMResponse after streaming."""
        if not self._finished:
            # Consume remaining chunks
            for _ in self:
                pass

        final_usage = None
        finish_reason = None
        for chunk in reversed(self._chunks):
            if chunk.usage:
                final_usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if final_usage and finish_reason:
                break

        return LLMResponse(
            content=self._content,
            model=self._model,
            provider=self._provider,
            usage=final_usage or Usage(),
            finish_reason=finish_reason,
        )


@dataclass
class CompletionRequest:
    """Request for LLM completion."""

    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def simple(
        cls,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> "CompletionRequest":
        """Create a simple completion request."""
        messages = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(prompt))
        return cls(messages=messages, **kwargs)
