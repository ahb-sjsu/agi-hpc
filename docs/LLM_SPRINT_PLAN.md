# LLM (Shared LLM Infrastructure) Sprint Plan

## Overview

The LLM subsystem provides a unified interface for language model access across the AGI-HPC cognitive architecture. It abstracts multiple LLM providers (OpenAI, Anthropic, local models) behind a consistent API, enabling the LH Planner, Metacognition, and other components to leverage language models for reasoning, planning, and reflection.

## Current State Assessment

### Implemented (Scaffolding)
| Component | Status | Location |
|-----------|--------|----------|
| `lh/llm/` directory | **Exists** | `src/agi/lh/llm/` |
| LLM adapter stubs | **Stub** | Basic interface defined |
| Core LLM client | **TODO** | `src/agi/core/llm/` mentioned in architecture |

### Key Gaps
1. **No unified client** - Each service implements own LLM access
2. **No provider abstraction** - Hardcoded to specific providers
3. **No streaming support** - Only synchronous completions
4. **No caching** - Repeated identical requests
5. **No rate limiting** - Risk of API quota exhaustion
6. **No fallback chain** - Single point of failure
7. **No prompt management** - Prompts scattered in code
8. **No token counting** - No budget management
9. **No observability** - No latency/cost tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        LLM CLIENT                                    │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │  complete()  │  │   stream()   │  │   embed()    │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   chat()     │  │   json()     │  │  function()  │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                       MIDDLEWARE STACK                               │   │
│   │                                                                      │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│   │   │  Cache   │──│ RateLimit│──│  Retry   │──│ Fallback │           │   │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘           │   │
│   │         │             │             │             │                  │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│   │   │ Metrics  │──│ Tracing  │──│ Logging  │──│  Budget  │           │   │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘           │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                         PROVIDERS                                    │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   Anthropic  │  │    OpenAI    │  │    Ollama    │             │   │
│   │   │   (Claude)   │  │  (GPT-4/o1)  │  │   (local)    │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │    vLLM      │  │  Together    │  │   Groq       │             │   │
│   │   │  (self-host) │  │   (cloud)    │  │   (fast)     │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases in AGI-HPC

| Component | Use Case | Model Requirements |
|-----------|----------|-------------------|
| LH Planner | Plan generation from goals | Strong reasoning, long context |
| LH Planner | Tool selection | Function calling support |
| Metacognition | Plan critique | Instruction following |
| Metacognition | Revision suggestions | Creative problem solving |
| Safety | Ethical reasoning fallback | Alignment, instruction following |
| Memory | Semantic search embedding | Text embeddings |
| Memory | Memory summarization | Compression, extraction |

---

## Sprint 1: Core LLM Client

**Goal**: Implement unified LLM client with provider abstraction.

### Tasks

#### 1.1 Provider Protocol

```python
# src/agi/core/llm/providers/base.py
"""Base LLM provider protocol."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class Role(Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """A chat message."""

    role: Role
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class FunctionDefinition:
    """A function/tool definition."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    strict: bool = False


@dataclass
class CompletionRequest:
    """Request for LLM completion."""

    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[str] = None  # "auto", "none", or function name
    json_mode: bool = False
    stream: bool = False
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    finish_reason: str  # "stop", "length", "function_call", "tool_calls"
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float = 0
    cached: bool = False


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    delta: str
    finish_reason: Optional[str] = None
    function_call_delta: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingRequest:
    """Request for text embedding."""

    texts: List[str]
    model: Optional[str] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding request."""

    embeddings: List[List[float]]
    model: str
    usage: Optional[Dict[str, int]] = None
    dimension: int = 0


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        ...

    @property
    def default_model(self) -> str:
        """Default model for this provider."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether streaming is supported."""
        ...

    @property
    def supports_functions(self) -> bool:
        """Whether function calling is supported."""
        ...

    @property
    def supports_json_mode(self) -> bool:
        """Whether JSON mode is supported."""
        ...

    async def complete(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion."""
        ...

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion."""
        ...

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        ...

    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text."""
        ...
```

#### 1.2 Anthropic Provider

```python
# src/agi/core/llm/providers/anthropic.py
"""Anthropic Claude provider."""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Optional

import anthropic

from agi.core.llm.providers.base import (
    LLMProvider,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Message,
    Role,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize Anthropic provider."""
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._base_url = base_url
        self._default_model = default_model

        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return True  # Via tool_use

    @property
    def supports_json_mode(self) -> bool:
        return True

    async def complete(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate completion with Claude."""
        start = time.time()

        # Extract system message
        system = None
        messages = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            else:
                messages.append({
                    "role": self._map_role(msg.role),
                    "content": msg.content,
                })

        # Build request kwargs
        kwargs = {
            "model": request.model or self._default_model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if request.temperature != 0.7:
            kwargs["temperature"] = request.temperature

        if request.stop:
            kwargs["stop_sequences"] = request.stop

        # Handle tool/function calling
        if request.functions:
            kwargs["tools"] = [
                {
                    "name": f.name,
                    "description": f.description,
                    "input_schema": f.parameters,
                }
                for f in request.functions
            ]

        try:
            response = await self._client.messages.create(**kwargs)

            content = ""
            tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    })

            return CompletionResponse(
                content=content,
                model=response.model,
                finish_reason=self._map_stop_reason(response.stop_reason),
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": (
                        response.usage.input_tokens +
                        response.usage.output_tokens
                    ),
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            raise

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion with Claude."""
        # Extract system and messages (same as complete)
        system = None
        messages = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system = msg.content
            else:
                messages.append({
                    "role": self._map_role(msg.role),
                    "content": msg.content,
                })

        kwargs = {
            "model": request.model or self._default_model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(delta=event.delta.text)
                elif event.type == "message_stop":
                    yield StreamChunk(delta="", finish_reason="stop")

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings (not natively supported, use Voyage)."""
        raise NotImplementedError(
            "Anthropic does not provide embeddings. Use voyage-3 via Voyage AI."
        )

    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens using Anthropic's tokenizer."""
        return await self._client.count_tokens(text)

    def _map_role(self, role: Role) -> str:
        """Map role to Anthropic format."""
        if role == Role.USER:
            return "user"
        elif role == Role.ASSISTANT:
            return "assistant"
        elif role == Role.TOOL:
            return "user"  # Tool results go as user messages
        else:
            return "user"

    def _map_stop_reason(self, reason: str) -> str:
        """Map Anthropic stop reason to standard."""
        if reason == "end_turn":
            return "stop"
        elif reason == "max_tokens":
            return "length"
        elif reason == "tool_use":
            return "tool_calls"
        else:
            return reason or "stop"
```

#### 1.3 OpenAI Provider

```python
# src/agi/core/llm/providers/openai.py
"""OpenAI GPT provider."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncIterator, Optional

import openai

from agi.core.llm.providers.base import (
    LLMProvider,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Role,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        default_model: str = "gpt-4o",
    ) -> None:
        """Initialize OpenAI provider."""
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._organization = organization
        self._default_model = default_model

        self._client = openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    async def complete(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate completion with GPT."""
        start = time.time()

        messages = [
            {
                "role": self._map_role(msg.role),
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
            }
            for msg in request.messages
        ]

        kwargs = {
            "model": request.model or self._default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if request.stop:
            kwargs["stop"] = request.stop

        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if request.functions:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": f.name,
                        "description": f.description,
                        "parameters": f.parameters,
                        **({"strict": True} if f.strict else {}),
                    },
                }
                for f in request.functions
            ]

            if request.function_call:
                if request.function_call == "auto":
                    kwargs["tool_choice"] = "auto"
                elif request.function_call == "none":
                    kwargs["tool_choice"] = "none"
                else:
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.function_call},
                    }

        if request.user:
            kwargs["user"] = request.user

        try:
            response = await self._client.chat.completions.create(**kwargs)

            choice = response.choices[0]

            return CompletionResponse(
                content=choice.message.content or "",
                model=response.model,
                finish_reason=choice.finish_reason,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (choice.message.tool_calls or [])
                ] if choice.message.tool_calls else None,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except openai.APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion with GPT."""
        messages = [
            {"role": self._map_role(msg.role), "content": msg.content}
            for msg in request.messages
        ]

        kwargs = {
            "model": request.model or self._default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    yield StreamChunk(
                        delta=delta.content or "",
                        finish_reason=finish_reason,
                    )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings with OpenAI."""
        model = request.model or "text-embedding-3-small"

        response = await self._client.embeddings.create(
            model=model,
            input=request.texts,
        )

        return EmbeddingResponse(
            embeddings=[e.embedding for e in response.data],
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            dimension=len(response.data[0].embedding) if response.data else 0,
        )

    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens using tiktoken."""
        import tiktoken

        model = model or self._default_model
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def _map_role(self, role: Role) -> str:
        """Map role to OpenAI format."""
        return role.value
```

#### 1.4 Ollama Provider (Local Models)

```python
# src/agi/core/llm/providers/ollama.py
"""Ollama local model provider."""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Optional

import httpx

from agi.core.llm.providers.base import (
    LLMProvider,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Role,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b",
    ) -> None:
        """Initialize Ollama provider."""
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.AsyncClient(timeout=300.0)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return False  # Limited support

    @property
    def supports_json_mode(self) -> bool:
        return True

    async def complete(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate completion with Ollama."""
        start = time.time()

        messages = [
            {
                "role": self._map_role(msg.role),
                "content": msg.content,
            }
            for msg in request.messages
        ]

        payload = {
            "model": request.model or self._default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }

        if request.json_mode:
            payload["format"] = "json"

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            content=data["message"]["content"],
            model=data["model"],
            finish_reason="stop" if data.get("done") else "length",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) +
                    data.get("eval_count", 0)
                ),
            },
            latency_ms=(time.time() - start) * 1000,
        )

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion with Ollama."""
        messages = [
            {"role": self._map_role(msg.role), "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": request.model or self._default_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)

                    if "message" in data:
                        yield StreamChunk(
                            delta=data["message"].get("content", ""),
                            finish_reason="stop" if data.get("done") else None,
                        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings with Ollama."""
        model = request.model or "nomic-embed-text"
        embeddings = []

        for text in request.texts:
            response = await self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            dimension=len(embeddings[0]) if embeddings else 0,
        )

    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Estimate token count."""
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def _map_role(self, role: Role) -> str:
        """Map role to Ollama format."""
        if role == Role.SYSTEM:
            return "system"
        elif role == Role.USER:
            return "user"
        elif role == Role.ASSISTANT:
            return "assistant"
        else:
            return "user"
```

#### 1.5 Unified LLM Client

```python
# src/agi/core/llm/client.py
"""Unified LLM client with provider routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

from agi.core.llm.providers.base import (
    LLMProvider,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Message,
    Role,
    FunctionDefinition,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    """LLM client configuration."""

    default_provider: str = "anthropic"
    default_model: Optional[str] = None
    fallback_providers: List[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600


class LLMClient:
    """Unified LLM client with provider abstraction.

    Features:
    - Multiple provider support (Anthropic, OpenAI, Ollama)
    - Automatic fallback on failure
    - Response caching
    - Token counting
    - Streaming support
    """

    def __init__(
        self,
        config: LLMClientConfig = None,
        providers: Dict[str, LLMProvider] = None,
    ) -> None:
        """Initialize LLM client."""
        self.config = config or LLMClientConfig()
        self._providers: Dict[str, LLMProvider] = providers or {}

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
    ) -> None:
        """Register an LLM provider."""
        self._providers[name] = provider
        logger.info("Registered LLM provider: %s", name)

    def get_provider(
        self,
        name: Optional[str] = None,
    ) -> LLMProvider:
        """Get a provider by name."""
        name = name or self.config.default_provider

        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")

        return self._providers[name]

    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        functions: Optional[List[FunctionDefinition]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion."""
        request = CompletionRequest(
            messages=messages,
            model=model or self.config.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            functions=functions,
            json_mode=json_mode,
            **kwargs,
        )

        # Try primary provider
        providers_to_try = [provider or self.config.default_provider]
        providers_to_try.extend(self.config.fallback_providers)

        last_error = None

        for provider_name in providers_to_try:
            try:
                llm = self.get_provider(provider_name)
                response = await llm.complete(request)

                logger.debug(
                    "Completion from %s: %d tokens in %.0fms",
                    provider_name,
                    response.usage.get("total_tokens", 0) if response.usage else 0,
                    response.latency_ms,
                )

                return response

            except Exception as e:
                logger.warning(
                    "Provider %s failed: %s",
                    provider_name, e,
                )
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion."""
        request = CompletionRequest(
            messages=messages,
            model=model or self.config.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        llm = self.get_provider(provider)

        async for chunk in llm.stream(request):
            yield chunk

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        request = EmbeddingRequest(
            texts=texts,
            model=model,
        )

        llm = self.get_provider(provider)
        return await llm.embed(request)

    async def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Simple chat interface."""
        messages = []

        if system_message:
            messages.append(Message(role=Role.SYSTEM, content=system_message))

        messages.append(Message(role=Role.USER, content=user_message))

        response = await self.complete(messages, **kwargs)
        return response.content

    async def json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """Get JSON response."""
        import json

        if schema:
            prompt += f"\n\nRespond with JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"

        response = await self.chat(prompt, json_mode=True, **kwargs)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s", e)
            raise

    async def function_call(
        self,
        messages: List[Message],
        functions: List[FunctionDefinition],
        function_choice: str = "auto",
        **kwargs,
    ) -> CompletionResponse:
        """Call with functions."""
        return await self.complete(
            messages=messages,
            functions=functions,
            function_call=function_choice,
            **kwargs,
        )
```

#### 1.6 Configuration

```yaml
# configs/llm_config.yaml
llm:
  default_provider: "anthropic"
  default_model: null  # Use provider's default

  fallback_providers:
    - "openai"
    - "ollama"

  timeout_seconds: 60
  max_retries: 3

  cache:
    enabled: true
    backend: "redis"  # memory, redis, disk
    ttl_seconds: 3600
    max_size_mb: 100

  providers:
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      default_model: "claude-sonnet-4-20250514"
      models:
        - "claude-sonnet-4-20250514"
        - "claude-3-5-haiku-20241022"
        - "claude-opus-4-20250514"

    openai:
      api_key: "${OPENAI_API_KEY}"
      default_model: "gpt-4o"
      models:
        - "gpt-4o"
        - "gpt-4o-mini"
        - "o1-preview"
        - "o1-mini"

    ollama:
      base_url: "http://localhost:11434"
      default_model: "llama3.1:8b"
      models:
        - "llama3.1:8b"
        - "llama3.1:70b"
        - "codellama:13b"
        - "mistral:7b"

  rate_limits:
    anthropic:
      requests_per_minute: 60
      tokens_per_minute: 100000
    openai:
      requests_per_minute: 500
      tokens_per_minute: 150000

  budget:
    enabled: true
    daily_limit_usd: 100.0
    alert_threshold_pct: 80

  monitoring:
    prometheus:
      enabled: true
    tracing:
      enabled: true
```

### Acceptance Criteria
```bash
# Test LLM client
python -c "
import asyncio
from agi.core.llm.client import LLMClient
from agi.core.llm.providers.anthropic import AnthropicProvider
from agi.core.llm.providers.base import Message, Role

async def test():
    client = LLMClient()
    client.register_provider('anthropic', AnthropicProvider())

    response = await client.chat(
        'What is 2 + 2?',
        system_message='You are a helpful assistant.',
    )
    print(f'Response: {response}')

asyncio.run(test())
"
```

---

## Sprint 2: Middleware Stack

**Goal**: Implement caching, rate limiting, retry, and fallback middleware.

### Tasks

#### 2.1 Caching Middleware

```python
# src/agi/core/llm/middleware/cache.py
"""Response caching middleware."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from agi.core.llm.providers.base import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response."""

    response: CompletionResponse
    created_at: float
    expires_at: float
    hits: int = 0


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        ...

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size

    async def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._cache.get(key)
        if entry and time.time() < entry.expires_at:
            entry.hits += 1
            return entry
        elif entry:
            del self._cache[key]
        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        self._cache[key] = entry

    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    async def clear(self) -> None:
        self._cache.clear()

    def _evict_oldest(self) -> None:
        oldest = min(self._cache.items(), key=lambda x: x[1].created_at)
        del self._cache[oldest[0]]


class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        import redis.asyncio as redis
        self._client = redis.from_url(url)

    async def get(self, key: str) -> Optional[CacheEntry]:
        data = await self._client.get(f"llm:cache:{key}")
        if data:
            return self._deserialize(data)
        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        ttl = int(entry.expires_at - time.time())
        if ttl > 0:
            await self._client.setex(
                f"llm:cache:{key}",
                ttl,
                self._serialize(entry),
            )

    async def delete(self, key: str) -> None:
        await self._client.delete(f"llm:cache:{key}")

    async def clear(self) -> None:
        keys = await self._client.keys("llm:cache:*")
        if keys:
            await self._client.delete(*keys)

    def _serialize(self, entry: CacheEntry) -> bytes:
        return json.dumps({
            "response": entry.response.__dict__,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "hits": entry.hits,
        }).encode()

    def _deserialize(self, data: bytes) -> CacheEntry:
        d = json.loads(data.decode())
        return CacheEntry(
            response=CompletionResponse(**d["response"]),
            created_at=d["created_at"],
            expires_at=d["expires_at"],
            hits=d["hits"],
        )


class CacheMiddleware:
    """Caching middleware for LLM requests."""

    def __init__(
        self,
        backend: CacheBackend,
        ttl_seconds: int = 3600,
        cacheable_temp_threshold: float = 0.3,
    ) -> None:
        """Initialize cache middleware."""
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self.cacheable_temp_threshold = cacheable_temp_threshold
        self._stats = {"hits": 0, "misses": 0}

    def cache_key(self, request: CompletionRequest) -> str:
        """Generate cache key for request."""
        # Include relevant request fields
        key_data = {
            "messages": [(m.role.value, m.content) for m in request.messages],
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "json_mode": request.json_mode,
        }

        if request.functions:
            key_data["functions"] = [f.name for f in request.functions]

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def is_cacheable(self, request: CompletionRequest) -> bool:
        """Check if request is cacheable."""
        # Don't cache high-temperature requests
        if request.temperature > self.cacheable_temp_threshold:
            return False

        # Don't cache streaming requests
        if request.stream:
            return False

        return True

    async def get_cached(
        self,
        request: CompletionRequest,
    ) -> Optional[CompletionResponse]:
        """Get cached response if available."""
        if not self.is_cacheable(request):
            return None

        key = self.cache_key(request)
        entry = await self.backend.get(key)

        if entry:
            self._stats["hits"] += 1
            logger.debug("Cache hit for key %s", key[:16])
            entry.response.cached = True
            return entry.response

        self._stats["misses"] += 1
        return None

    async def cache_response(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
    ) -> None:
        """Cache a response."""
        if not self.is_cacheable(request):
            return

        key = self.cache_key(request)
        now = time.time()

        entry = CacheEntry(
            response=response,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )

        await self.backend.set(key, entry)
        logger.debug("Cached response for key %s", key[:16])

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            **self._stats,
            "hit_rate": hit_rate,
        }
```

#### 2.2 Rate Limiting Middleware

```python
# src/agi/core/llm/middleware/rate_limit.py
"""Rate limiting middleware."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
    ) -> None:
        """Initialize token bucket."""
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, return wait time if needed."""
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate

            return wait_time

    async def acquire_wait(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            logger.debug("Rate limit: waiting %.2fs", wait_time)
            await asyncio.sleep(wait_time)
            await self.acquire_wait(tokens)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now


class RateLimiter:
    """Rate limiter for LLM providers."""

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self._limiters: Dict[str, Dict[str, TokenBucket]] = {}

    def configure(
        self,
        provider: str,
        config: RateLimitConfig,
    ) -> None:
        """Configure rate limits for a provider."""
        self._limiters[provider] = {
            "requests": TokenBucket(
                capacity=config.requests_per_minute,
                refill_rate=config.requests_per_minute / 60.0,
            ),
            "tokens": TokenBucket(
                capacity=config.tokens_per_minute,
                refill_rate=config.tokens_per_minute / 60.0,
            ),
        }

        logger.info(
            "Rate limit configured for %s: %d req/min, %d tok/min",
            provider,
            config.requests_per_minute,
            config.tokens_per_minute,
        )

    async def acquire_request(self, provider: str) -> None:
        """Acquire permission for a request."""
        if provider in self._limiters:
            await self._limiters[provider]["requests"].acquire_wait(1)

    async def acquire_tokens(
        self,
        provider: str,
        tokens: int,
    ) -> None:
        """Acquire permission for tokens."""
        if provider in self._limiters:
            await self._limiters[provider]["tokens"].acquire_wait(tokens)
```

#### 2.3 Retry Middleware

```python
# src/agi/core/llm/middleware/retry.py
"""Retry middleware with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional, Set, Type

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Set[Type[Exception]] = None

    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = {
                ConnectionError,
                TimeoutError,
            }


class RetryMiddleware:
    """Retry middleware with exponential backoff."""

    def __init__(self, config: RetryConfig = None) -> None:
        """Initialize retry middleware."""
        self.config = config or RetryConfig()

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        """Execute function with retries."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                if not self._is_retryable(e):
                    raise

                last_error = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        "Retry %d/%d after %.2fs: %s",
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)

        raise last_error

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        for exc_type in self.config.retryable_exceptions:
            if isinstance(error, exc_type):
                return True

        # Check for specific status codes
        if hasattr(error, "status_code"):
            return error.status_code in {429, 500, 502, 503, 504}

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        delay = min(
            self.config.max_delay,
            self.config.initial_delay * (
                self.config.exponential_base ** attempt
            ),
        )

        if self.config.jitter:
            delay *= random.uniform(0.5, 1.5)

        return delay
```

### Deliverables
- [ ] Memory cache backend
- [ ] Redis cache backend
- [ ] Request-based rate limiter
- [ ] Token-based rate limiter
- [ ] Retry with exponential backoff
- [ ] Provider fallback chain

---

## Sprint 3: Prompt Management

**Goal**: Implement prompt templates and versioning.

### Tasks

#### 3.1 Prompt Template System

```python
# src/agi/core/llm/prompts/template.py
"""Prompt template system."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agi.core.llm.providers.base import Message, Role

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with variables."""

    name: str
    version: str
    description: str = ""
    system: str = ""
    user: str = ""
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> List[Message]:
        """Render template with variables."""
        # Validate all required variables provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        messages = []

        # Render system message
        if self.system:
            system_content = self._substitute(self.system, kwargs)
            messages.append(Message(role=Role.SYSTEM, content=system_content))

        # Add examples (few-shot)
        for example in self.examples:
            if "user" in example:
                messages.append(Message(
                    role=Role.USER,
                    content=self._substitute(example["user"], kwargs),
                ))
            if "assistant" in example:
                messages.append(Message(
                    role=Role.ASSISTANT,
                    content=example["assistant"],
                ))

        # Render user message
        if self.user:
            user_content = self._substitute(self.user, kwargs)
            messages.append(Message(role=Role.USER, content=user_content))

        return messages

    def _substitute(self, template: str, values: Dict[str, Any]) -> str:
        """Substitute variables in template."""
        result = template

        for key, value in values.items():
            # Support both {{var}} and {var} syntax
            result = result.replace(f"{{{{{key}}}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))

        return result


class PromptRegistry:
    """Registry for prompt templates."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        if template.name not in self._templates:
            self._templates[template.name] = {}

        self._templates[template.name][template.version] = template

        logger.info(
            "Registered prompt template: %s v%s",
            template.name,
            template.version,
        )

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> PromptTemplate:
        """Get a template by name and optional version."""
        if name not in self._templates:
            raise ValueError(f"Unknown template: {name}")

        versions = self._templates[name]

        if version:
            if version not in versions:
                raise ValueError(f"Unknown version {version} for {name}")
            return versions[version]

        # Return latest version
        latest = max(versions.keys())
        return versions[latest]

    def load_directory(self, path: Path) -> int:
        """Load templates from YAML files in directory."""
        count = 0

        for file in path.glob("**/*.yaml"):
            with open(file) as f:
                data = yaml.safe_load(f)

            for template_data in data.get("templates", []):
                template = PromptTemplate(
                    name=template_data["name"],
                    version=template_data.get("version", "1.0"),
                    description=template_data.get("description", ""),
                    system=template_data.get("system", ""),
                    user=template_data.get("user", ""),
                    variables=template_data.get("variables", []),
                    examples=template_data.get("examples", []),
                    metadata=template_data.get("metadata", {}),
                )
                self.register(template)
                count += 1

        logger.info("Loaded %d prompt templates from %s", count, path)
        return count
```

#### 3.2 Prompt Library

```yaml
# prompts/planning.yaml
templates:
  - name: "plan_generation"
    version: "1.0"
    description: "Generate a hierarchical plan for a goal"
    variables:
      - goal
      - context
      - available_tools
    system: |
      You are a planning agent for an AGI system. Your task is to generate
      hierarchical plans that decompose high-level goals into executable steps.

      Guidelines:
      - Break goals into subgoals, then into atomic actions
      - Each action should map to an available tool
      - Consider dependencies between steps
      - Include error handling for risky operations
      - Be specific about parameters and expected outcomes

      Available tools:
      {{available_tools}}

    user: |
      Generate a plan for the following goal:

      Goal: {{goal}}

      Context: {{context}}

      Respond with a JSON plan structure:
      {
        "goal": "string",
        "subgoals": [
          {
            "description": "string",
            "steps": [
              {
                "action": "string",
                "tool": "string",
                "parameters": {},
                "expected_outcome": "string"
              }
            ]
          }
        ]
      }

    examples:
      - user: |
          Generate a plan for the following goal:

          Goal: Pick up the red cube and place it on the table

          Context: Robot is at home position. Red cube is on the floor at (1, 0, 0.1).
        assistant: |
          {
            "goal": "Pick up red cube and place on table",
            "subgoals": [
              {
                "description": "Navigate to cube location",
                "steps": [
                  {
                    "action": "Move arm to pre-grasp position",
                    "tool": "move_arm",
                    "parameters": {"target": [1, 0, 0.3]},
                    "expected_outcome": "Arm above cube"
                  }
                ]
              },
              {
                "description": "Grasp the cube",
                "steps": [
                  {
                    "action": "Lower arm to cube",
                    "tool": "move_arm",
                    "parameters": {"target": [1, 0, 0.1]},
                    "expected_outcome": "Gripper at cube level"
                  },
                  {
                    "action": "Close gripper",
                    "tool": "gripper",
                    "parameters": {"action": "close"},
                    "expected_outcome": "Cube grasped"
                  }
                ]
              }
            ]
          }

  - name: "metacognition_review"
    version: "1.0"
    description: "Review a plan for issues and improvements"
    variables:
      - plan
      - reasoning_trace
    system: |
      You are a metacognitive reviewer for an AGI system. Your role is to
      critically analyze plans and reasoning traces to identify:

      1. Logical inconsistencies
      2. Missing steps or dependencies
      3. Safety concerns
      4. Efficiency improvements
      5. Alternative approaches

      Be thorough but constructive.

    user: |
      Review the following plan and reasoning:

      Plan:
      {{plan}}

      Reasoning trace:
      {{reasoning_trace}}

      Provide your analysis in JSON format:
      {
        "decision": "ACCEPT" | "REVISE" | "REJECT",
        "confidence": 0.0-1.0,
        "issues": [
          {
            "severity": "info" | "warning" | "error" | "critical",
            "category": "string",
            "description": "string",
            "affected_steps": [],
            "suggested_fix": "string"
          }
        ],
        "suggestions": []
      }
```

### Deliverables
- [ ] Prompt template system
- [ ] Variable substitution
- [ ] Few-shot example injection
- [ ] Prompt versioning
- [ ] YAML-based prompt library
- [ ] Planning prompts
- [ ] Metacognition prompts

---

## Sprint 4: Observability

**Goal**: Implement comprehensive monitoring and tracing.

### Tasks

#### 4.1 Prometheus Metrics

```python
# src/agi/core/llm/metrics.py
"""LLM metrics for Prometheus."""

from prometheus_client import Counter, Histogram, Gauge

# Request counters
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens used",
    ["provider", "model", "type"],  # type: prompt, completion
)

# Latency histogram
LLM_REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency",
    ["provider", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Cost tracking
LLM_COST_USD = Counter(
    "llm_cost_usd_total",
    "Total cost in USD",
    ["provider", "model"],
)

# Cache metrics
LLM_CACHE_HITS = Counter(
    "llm_cache_hits_total",
    "Cache hits",
    [],
)

LLM_CACHE_MISSES = Counter(
    "llm_cache_misses_total",
    "Cache misses",
    [],
)

# Rate limit metrics
LLM_RATE_LIMIT_WAITS = Counter(
    "llm_rate_limit_waits_total",
    "Rate limit wait events",
    ["provider"],
)

LLM_RATE_LIMIT_WAIT_SECONDS = Histogram(
    "llm_rate_limit_wait_seconds",
    "Rate limit wait duration",
    ["provider"],
)

# Error tracking
LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["provider", "error_type"],
)

# Active requests gauge
LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently active requests",
    ["provider"],
)
```

#### 4.2 Cost Tracking

```python
# src/agi/core/llm/cost.py
"""LLM cost tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing for a model (per 1M tokens)."""

    prompt_usd: float
    completion_usd: float


# Pricing data (as of late 2024 - update regularly)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Anthropic
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0),
    "claude-3-5-haiku-20241022": ModelPricing(1.0, 5.0),
    "claude-opus-4-20250514": ModelPricing(15.0, 75.0),

    # OpenAI
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "o1-preview": ModelPricing(15.0, 60.0),
    "o1-mini": ModelPricing(3.0, 12.0),

    # Local (free)
    "llama3.1:8b": ModelPricing(0.0, 0.0),
    "llama3.1:70b": ModelPricing(0.0, 0.0),
}


class CostTracker:
    """Track LLM usage costs."""

    def __init__(self, daily_limit_usd: float = 0) -> None:
        """Initialize cost tracker."""
        self.daily_limit_usd = daily_limit_usd
        self._total_cost = 0.0
        self._daily_cost = 0.0
        self._by_model: Dict[str, float] = {}

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for a request."""
        pricing = MODEL_PRICING.get(model)

        if not pricing:
            logger.warning("No pricing for model: %s", model)
            return 0.0

        cost = (
            (prompt_tokens / 1_000_000) * pricing.prompt_usd +
            (completion_tokens / 1_000_000) * pricing.completion_usd
        )

        return cost

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Record usage and return cost."""
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        self._total_cost += cost
        self._daily_cost += cost
        self._by_model[model] = self._by_model.get(model, 0) + cost

        # Check budget
        if self.daily_limit_usd > 0 and self._daily_cost > self.daily_limit_usd:
            logger.warning(
                "Daily budget exceeded: $%.2f > $%.2f",
                self._daily_cost,
                self.daily_limit_usd,
            )

        return cost

    def reset_daily(self) -> None:
        """Reset daily counter."""
        self._daily_cost = 0.0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def daily_cost(self) -> float:
        return self._daily_cost

    @property
    def by_model(self) -> Dict[str, float]:
        return self._by_model.copy()
```

### Deliverables
- [ ] Prometheus metrics
- [ ] Cost tracking per model
- [ ] Budget alerts
- [ ] OpenTelemetry tracing
- [ ] Grafana dashboard

---

## Sprint 5: Unit Tests

**Goal**: Achieve 80%+ test coverage for LLM module.

### Tasks

#### 5.1 Provider Tests
- [ ] `test_anthropic_complete`
- [ ] `test_anthropic_stream`
- [ ] `test_openai_complete`
- [ ] `test_openai_stream`
- [ ] `test_openai_embed`
- [ ] `test_ollama_complete`

#### 5.2 Client Tests
- [ ] `test_client_complete`
- [ ] `test_client_chat`
- [ ] `test_client_json`
- [ ] `test_client_function_call`
- [ ] `test_client_fallback`

#### 5.3 Middleware Tests
- [ ] `test_cache_hit_miss`
- [ ] `test_rate_limiter`
- [ ] `test_retry_backoff`
- [ ] `test_fallback_chain`

#### 5.4 Prompt Tests
- [ ] `test_template_render`
- [ ] `test_variable_substitution`
- [ ] `test_registry_load`

---

## Sprint 6: Integration and Production

**Goal**: Integrate with AGI-HPC components and prepare for production.

### Tasks

#### 6.1 Component Integration
- [ ] LH Planner integration
- [ ] Metacognition integration
- [ ] Memory embedding generation
- [ ] Safety fallback

#### 6.2 Production Configuration
```yaml
# deploy/llm/production.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
  namespace: agi-hpc
data:
  config.yaml: |
    llm:
      default_provider: "anthropic"

      providers:
        anthropic:
          api_key: "${ANTHROPIC_API_KEY}"

      cache:
        backend: "redis"
        redis_url: "redis://redis:6379/1"

      rate_limits:
        anthropic:
          requests_per_minute: 100
          tokens_per_minute: 200000

      budget:
        enabled: true
        daily_limit_usd: 500.0
        alert_webhook: "${SLACK_WEBHOOK}"
```

---

## File Structure After Completion

```
src/agi/core/llm/
├── __init__.py
├── client.py               # Unified LLM client
├── config.py               # Configuration
├── providers/
│   ├── __init__.py
│   ├── base.py             # Provider protocol
│   ├── anthropic.py        # Anthropic Claude
│   ├── openai.py           # OpenAI GPT
│   ├── ollama.py           # Ollama local
│   ├── vllm.py             # vLLM self-hosted
│   └── together.py         # Together AI
├── middleware/
│   ├── __init__.py
│   ├── cache.py            # Response caching
│   ├── rate_limit.py       # Rate limiting
│   ├── retry.py            # Retry logic
│   └── fallback.py         # Provider fallback
├── prompts/
│   ├── __init__.py
│   ├── template.py         # Template system
│   └── registry.py         # Prompt registry
├── metrics.py              # Prometheus metrics
└── cost.py                 # Cost tracking

prompts/
├── planning.yaml
├── metacognition.yaml
├── memory.yaml
└── safety.yaml

tests/core/llm/
├── __init__.py
├── conftest.py
├── test_client.py
├── test_providers/
│   ├── test_anthropic.py
│   ├── test_openai.py
│   └── test_ollama.py
├── test_middleware/
│   ├── test_cache.py
│   ├── test_rate_limit.py
│   └── test_retry.py
└── test_prompts.py

configs/
└── llm_config.yaml
```

---

## Priority Order

1. **Sprint 1** - Critical: Core client enables all LLM usage
2. **Sprint 2** - High: Middleware for reliability
3. **Sprint 4** - High: Observability for operations
4. **Sprint 3** - Medium: Prompt management for maintainability
5. **Sprint 5** - High: Tests for correctness
6. **Sprint 6** - Medium: Integration

---

## Dependencies

```toml
# pyproject.toml additions for LLM
[project.optional-dependencies]
llm = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "tiktoken>=0.7.0",
    "redis>=5.0",
    "prometheus-client>=0.19",
    "pyyaml>=6.0",
]

llm-local = [
    "agi-hpc[llm]",
    "vllm>=0.6.0",
]
```
