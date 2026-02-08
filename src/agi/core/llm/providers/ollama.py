# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Ollama provider implementation.

Supports local LLM inference via Ollama for models like Llama, Mistral, etc.

Environment Variables:
    OLLAMA_HOST          Ollama server URL (default: http://localhost:11434)
    AGI_LLM_MODEL        Model to use (default: llama3.2)
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, List, Optional

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


class OllamaProvider(BaseProvider):
    """
    Provider for Ollama local LLM inference.

    Supports both the /api/generate and /api/chat endpoints.
    """

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        import os
        self._base_url = (
            self._config.base_url or
            os.getenv("OLLAMA_HOST") or
            self.DEFAULT_URL
        )

    @property
    def name(self) -> str:
        return "ollama"

    def _get_default_model(self) -> str:
        return "llama3.2"

    def _build_chat_payload(self, request: CompletionRequest) -> dict:
        """Build payload for /api/chat endpoint."""
        messages = []
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        return {
            "model": self._get_model(request),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._get_temperature(request),
                "num_predict": self._get_max_tokens(request),
            },
        }

    def _build_generate_payload(self, request: CompletionRequest) -> dict:
        """Build payload for /api/generate endpoint."""
        # Combine messages into a single prompt
        system_prompt = None
        prompt_parts = []

        for msg in request.messages:
            if msg.role.value == "system":
                system_prompt = msg.content
            elif msg.role.value == "user":
                prompt_parts.append(msg.content)
            elif msg.role.value == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self._get_model(request),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._get_temperature(request),
                "num_predict": self._get_max_tokens(request),
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        return payload

    def _complete_impl(self, request: CompletionRequest) -> LLMResponse:
        """Execute completion using chat endpoint."""
        url = f"{self._base_url}/api/chat"
        payload = self._build_chat_payload(request)

        if httpx is not None:
            return self._complete_httpx(url, payload)
        else:
            return self._complete_urllib(url, payload)

    def _complete_httpx(self, url: str, payload: dict) -> LLMResponse:
        """Complete using httpx library."""
        with httpx.Client(timeout=self._config.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()

        return self._parse_chat_response(result)

    def _complete_urllib(self, url: str, payload: dict) -> LLMResponse:
        """Complete using urllib (fallback)."""
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return self._parse_chat_response(result)

    def _parse_chat_response(self, result: dict) -> LLMResponse:
        """Parse chat endpoint response."""
        message = result.get("message", {})
        content = message.get("content", "")

        # Ollama provides token counts
        usage = Usage(
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_tokens=(
                result.get("prompt_eval_count", 0) +
                result.get("eval_count", 0)
            ),
        )

        return LLMResponse(
            content=content,
            model=result.get("model", ""),
            usage=usage,
            finish_reason="stop" if result.get("done") else None,
            raw=result,
        )

    def _stream_impl(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Execute streaming completion."""
        if httpx is None:
            raise RuntimeError("httpx is required for streaming")

        url = f"{self._base_url}/api/chat"
        payload = self._build_chat_payload(request)
        payload["stream"] = True

        content = ""
        with httpx.Client(timeout=self._config.timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
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
        message = event.get("message", {})
        text = message.get("content", "")
        done = event.get("done", False)

        if text or done:
            usage = None
            if done:
                usage = Usage(
                    prompt_tokens=event.get("prompt_eval_count", 0),
                    completion_tokens=event.get("eval_count", 0),
                )

            return StreamChunk(
                content=current_content + text,
                delta=text,
                finish_reason="stop" if done else None,
                is_final=done,
                usage=usage,
            )

        return None

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            url = f"{self._base_url}/api/tags"
            if httpx is not None:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(url)
                    return resp.status_code == 200
            else:
                import urllib.request
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug("[llm][ollama] Not available: %s", e)
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            url = f"{self._base_url}/api/tags"
            if httpx is not None:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(url)
                    result = resp.json()
            else:
                import urllib.request
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

            return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.error("[llm][ollama] Failed to list models: %s", e)
            return []
