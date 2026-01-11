# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Anthropic LLM Adapter for Claude API.

Provides integration with Anthropic's Claude models for planning.
"""

from __future__ import annotations

import json
import logging
from typing import Optional
import urllib.request
import urllib.error

from agi.lh.llm.adapter import BaseLLMAdapter, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude API.

    API Docs: https://docs.anthropic.com/en/api/messages

    Requires:
        AGI_LH_LLM_API_KEY: Anthropic API key
        AGI_LH_LLM_MODEL: e.g., claude-3-5-sonnet-20241022
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if not self._config.api_key:
            logger.warning(
                "[LLM][Anthropic] No API key provided. "
                "Set AGI_LH_LLM_API_KEY environment variable."
            )
        # Default to Claude 3.5 Sonnet if not specified
        if self._config.model == "llama3.2":  # Default from base config
            self._config.model = "claude-3-5-sonnet-20241022"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Anthropic Messages API.
        """
        if not self._config.api_key:
            raise ValueError("Anthropic API key not configured")

        payload = {
            "model": self._config.model,
            "max_tokens": self._get_max_tokens(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add temperature if not default
        temp = self._get_temperature(temperature)
        if temp != 1.0:
            payload["temperature"] = temp

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.API_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._config.api_key,
                    "anthropic-version": self.API_VERSION,
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Extract content from response
            content = ""
            if result.get("content"):
                content = "".join(
                    block.get("text", "")
                    for block in result["content"]
                    if block.get("type") == "text"
                )

            usage = result.get("usage", {})
            return LLMResponse(
                content=content,
                model=result.get("model", self._config.model),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": (
                        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    ),
                },
                raw=result,
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error(f"[LLM][Anthropic] HTTP {e.code}: {error_body}")
            raise ConnectionError(
                f"Anthropic API error: {e.code} - {error_body}"
            ) from e
        except urllib.error.URLError as e:
            logger.error(f"[LLM][Anthropic] Request failed: {e}")
            raise ConnectionError(f"Anthropic API error: {e}") from e

    def is_available(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self._config.api_key:
            return False

        # Simple check - try to make a minimal request
        # In practice, we just verify we have an API key
        return True
