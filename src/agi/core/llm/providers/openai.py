# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
OpenAI provider implementation.

Supports GPT-4, GPT-4o, and other OpenAI models via the Chat Completions API.

Environment Variables:
    OPENAI_API_KEY       API key for authentication
    OPENAI_BASE_URL      Base URL (for Azure or other compatible APIs)
    AGI_LLM_MODEL        Model to use (default: gpt-4o)
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


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI Chat Completions API.

    Also compatible with Azure OpenAI and other OpenAI-compatible APIs.
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        import os
        self._api_key = self._config.api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = (
            self._config.base_url or
            os.getenv("OPENAI_BASE_URL") or
            self.API_URL
        )

        if not self._api_key:
            logger.warning("[llm][openai] No API key. Set OPENAI_API_KEY.")

    @property
    def name(self) -> str:
        return "openai"

    def _get_default_model(self) -> str:
        return "gpt-4o"

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key or ''}",
        }

    def _build_payload(self, request: CompletionRequest) -> dict:
        """Build the API request payload."""
        messages = [msg.to_dict() for msg in request.messages]

        payload = {
            "model": self._get_model(request),
            "messages": messages,
            "max_tokens": self._get_max_tokens(request),
            "temperature": self._get_temperature(request),
        }

        if request.stop:
            payload["stop"] = request.stop

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty

        return payload

    def _complete_impl(self, request: CompletionRequest) -> LLMResponse:
        """Execute completion using httpx or urllib."""
        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

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
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        usage_data = result.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            model=result.get("model", ""),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            raw=result,
        )

    def _stream_impl(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Execute streaming completion."""
        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

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
                        yield StreamChunk(
                            content=content,
                            delta="",
                            is_final=True,
                        )
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
        choices = event.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        text = delta.get("content", "")
        if text or finish_reason:
            return StreamChunk(
                content=current_content + text,
                delta=text,
                finish_reason=finish_reason,
                is_final=finish_reason is not None,
            )

        return None

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        return bool(self._api_key)
