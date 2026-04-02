# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Async OpenAI-compatible LLM client for AGI-HPC.

Wraps llama-server (or any OpenAI-compatible endpoint) with:
- Async HTTP via aiohttp
- Streaming support
- Structured CompletionResult
- Latency tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from agi.meta.llm.config import InferenceConfig  # noqa: E402


@dataclass
class CompletionResult:
    """Result of a single LLM completion call.

    Attributes:
        text: Generated text content.
        tokens_used: Total tokens consumed (prompt + completion).
        latency_ms: Wall-clock latency in milliseconds.
        model: Model identifier from the server response.
        prompt_tokens: Tokens used by the prompt.
        completion_tokens: Tokens generated.
        finish_reason: Why generation stopped (stop, length, etc).
    """

    text: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""


class LLMClient:
    """Async OpenAI-compatible LLM client.

    Connects to llama-server or any endpoint that implements the
    ``/v1/chat/completions`` API.

    Usage::

        client = LLMClient(base_url="http://localhost:8080")
        result = await client.generate("Explain Paxos consensus.")
        print(result.text)

    Args:
        base_url: Base URL of the OpenAI-compatible server.
        timeout: Request timeout in seconds.
        default_model: Model name to send in requests.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 300.0,
        default_model: str = "default",
    ) -> None:
        if aiohttp is None:
            raise RuntimeError(
                "aiohttp is required but not installed. "
                "Install with: pip install aiohttp"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-create and return the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> CompletionResult:
        """Generate a completion from the LLM.

        Either provide a simple ``prompt`` string or a full ``messages``
        list.  When ``prompt`` is given it is wrapped as a single user
        message.  A system prompt from *config* (if any) is prepended.

        Args:
            prompt: Simple user prompt string.
            config: Inference configuration (temperature, etc.).
            messages: Override full message list.

        Returns:
            CompletionResult with generated text and metadata.
        """
        config = config or InferenceConfig()
        session = await self._get_session()

        if messages is None:
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "messages": messages,
            "stream": False,
            **config.to_api_params(),
        }
        if not body.get("model"):
            body["model"] = self._default_model

        url = f"{self._base_url}/v1/chat/completions"
        t0 = time.perf_counter()

        try:
            async with session.post(url, json=body) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            logger.exception("[llm-client] request failed url=%s", url)
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})

        return CompletionResult(
            text=choice.get("message", {}).get("content", ""),
            tokens_used=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            model=data.get("model", self._default_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM.

        Yields individual text deltas as they arrive from the server.

        Args:
            prompt: Simple user prompt string.
            config: Inference configuration.
            messages: Override full message list.

        Yields:
            Text deltas from the streaming response.
        """
        config = config or InferenceConfig()
        session = await self._get_session()

        if messages is None:
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "messages": messages,
            "stream": True,
            **config.to_api_params(),
        }
        if not body.get("model"):
            body["model"] = self._default_model

        url = f"{self._base_url}/v1/chat/completions"

        async with session.post(url, json=body) as resp:
            resp.raise_for_status()
            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> bool:
        """Check if the LLM endpoint is reachable.

        Returns:
            True if the server responds to /health.
        """
        session = await self._get_session()
        try:
            async with session.get(f"{self._base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False
