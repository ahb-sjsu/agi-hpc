# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Ollama LLM Adapter for local model inference.

Ollama provides local LLM inference with models like Llama, Mistral, etc.
This adapter interfaces with the Ollama HTTP API.
"""

from __future__ import annotations

import json
import logging
from typing import Optional
import urllib.request
import urllib.error

from agi.lh.llm.adapter import BaseLLMAdapter, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseLLMAdapter):
    """
    Adapter for Ollama local LLM inference.

    Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

    Default endpoint: http://localhost:11434
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._base_url = self._config.base_url or "http://localhost:11434"
        logger.info(f"[LLM][Ollama] Using endpoint: {self._base_url}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Ollama API.

        Uses the /api/generate endpoint for simple completion.
        """
        url = f"{self._base_url}/api/generate"

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._get_temperature(temperature),
                "num_predict": self._get_max_tokens(max_tokens),
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            return LLMResponse(
                content=result.get("response", ""),
                model=result.get("model", self._config.model),
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": (
                        result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    ),
                },
                raw=result,
            )

        except urllib.error.URLError as e:
            logger.error(f"[LLM][Ollama] Request failed: {e}")
            raise ConnectionError(f"Ollama API error: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"[LLM][Ollama] Invalid JSON response: {e}")
            raise ValueError(f"Invalid Ollama response: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"[LLM][Ollama] Not available: {e}")
            return False

    def list_models(self) -> list:
        """List available models in Ollama."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.error(f"[LLM][Ollama] Failed to list models: {e}")
            return []
