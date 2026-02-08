# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Anthropic Claude provider implementation.

Supports Claude 3.x and Claude 4 models via the Messages API.

Environment Variables:
    ANTHROPIC_API_KEY    API key for authentication
    AGI_LLM_MODEL        Model to use (default: claude-3-5-sonnet-20241022)
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional

from agi.core.llm.config import LLMConfig
from agi.core.llm.providers.base import BaseProvider
from agi.core.llm.types import (
    CompletionRequest,
    LLMResponse,
    StreamChunk,
    Usage,
)

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude API.

    Uses the Messages API: https://docs.anthropic.com/en/api/messages
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        import os
        self._api_key = self._config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._base_url = self._config.base_url or self.API_URL

        if not self._api_key:
            logger.warning(
                "[llm][anthropic] No API key. Set ANTHROPIC_API_KEY."
            )

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key or "",
            "anthropic-version": self.API_VERSION,
        }

    def _build_payload(self, request: CompletionRequest) -> dict:
        """Build the API request payload."""
        messages = []
        system_prompt = None

        for msg in request.messages:
            if msg.role.value == "system":
                system_prompt = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        payload = {
            "model": self._get_model(request),
            "max_tokens": self._get_max_tokens(request),
            "messages": messages,
        }

        temp = self._get_temperature(request)
        if temp != 1.0:
            payload["temperature"] = temp

        if system_prompt:
            payload["system"] = system_prompt

        if request.stop:
            payload["stop_sequences"] = request.stop

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        return payload

    def _complete_impl(self, request: CompletionRequest) -> LLMResponse:
        """Execute completion using httpx or urllib."""
        if not self._api_key:
            raise ValueError("Anthropic API key not configured")

        payload = self._build_payload(request)
        headers = self._build_headers()

        if httpx is not None:
            return self._complete_httpx(payload, headers)
        else:
            return self._complete_urllib(payload, headers)

    def _complete_httpx(self, payload: dict, headers: dict) -> LLMResponse:
        """Complete using httpx library."""
        with httpx.Client(timeout=self._config.timeout) as client:
            resp = client.post(self._base_url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        return self._parse_response(result)

    def _complete_urllib(self, payload: dict, headers: dict) -> LLMResponse:
        """Complete using urllib (fallback)."""
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url,
            data=data,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return self._parse_response(result)

    def _parse_response(self, result: dict) -> LLMResponse:
        """Parse the API response into LLMResponse."""
        content = ""
        if result.get("content"):
            content = "".join(
                block.get("text", "")
                for block in result["content"]
                if block.get("type") == "text"
            )

        usage_data = result.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(
                usage_data.get("input_tokens", 0) +
                usage_data.get("output_tokens", 0)
            ),
        )

        return LLMResponse(
            content=content,
            model=result.get("model", ""),
            usage=usage,
            finish_reason=result.get("stop_reason"),
            raw=result,
        )

    def _stream_impl(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Execute streaming completion."""
        if not self._api_key:
            raise ValueError("Anthropic API key not configured")

        if httpx is None:
            raise RuntimeError("httpx is required for streaming")

        payload = self._build_payload(request)
        payload["stream"] = True
        headers = self._build_headers()

        content = ""
        with httpx.Client(timeout=self._config.timeout) as client:
            with client.stream(
                "POST",
                self._base_url,
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk = self._parse_stream_event(event, content)
                    if chunk:
                        content = chunk.content
                        yield chunk

    def _parse_stream_event(
        self, event: dict, current_content: str
    ) -> Optional[StreamChunk]:
        """Parse a stream event into a StreamChunk."""
        event_type = event.get("type")

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                return StreamChunk(
                    content=current_content + text,
                    delta=text,
                )

        elif event_type == "message_stop":
            return StreamChunk(
                content=current_content,
                delta="",
                is_final=True,
                finish_reason="stop",
            )

        elif event_type == "message_delta":
            usage = event.get("usage", {})
            if usage:
                return StreamChunk(
                    content=current_content,
                    delta="",
                    usage=Usage(
                        prompt_tokens=usage.get("input_tokens", 0),
                        completion_tokens=usage.get("output_tokens", 0),
                    ),
                )

        return None

    def is_available(self) -> bool:
        """Check if Anthropic API is accessible."""
        return bool(self._api_key)
