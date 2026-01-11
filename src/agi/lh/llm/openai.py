# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
OpenAI LLM Adapter for GPT API.

Provides integration with OpenAI's GPT models for planning.
"""

from __future__ import annotations

import json
import logging
from typing import Optional
import urllib.request
import urllib.error

from agi.lh.llm.adapter import BaseLLMAdapter, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI Chat Completions API.

    API Docs: https://platform.openai.com/docs/api-reference/chat

    Requires:
        AGI_LH_LLM_API_KEY: OpenAI API key
        AGI_LH_LLM_MODEL: e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if not self._config.api_key:
            logger.warning(
                "[LLM][OpenAI] No API key provided. "
                "Set AGI_LH_LLM_API_KEY environment variable."
            )
        # Default to GPT-4o if not specified
        if self._config.model == "llama3.2":  # Default from base config
            self._config.model = "gpt-4o"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using OpenAI Chat Completions API.
        """
        if not self._config.api_key:
            raise ValueError("OpenAI API key not configured")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._get_max_tokens(max_tokens),
            "temperature": self._get_temperature(temperature),
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.API_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Extract content from response
            content = ""
            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            usage = result.get("usage", {})
            return LLMResponse(
                content=content,
                model=result.get("model", self._config.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                raw=result,
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error(f"[LLM][OpenAI] HTTP {e.code}: {error_body}")
            raise ConnectionError(f"OpenAI API error: {e.code} - {error_body}") from e
        except urllib.error.URLError as e:
            logger.error(f"[LLM][OpenAI] Request failed: {e}")
            raise ConnectionError(f"OpenAI API error: {e}") from e

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self._config.api_key:
            return False
        # Simple check - we have an API key
        return True
