"""
Thin OpenRouter client for AGI-HPC.

Used by Planner, MemoryClient, SafetyClient, and MetacognitionClient
to call LLMs for planning, reflection, or semantic extraction.

Requires:
    export OPENROUTER_API_KEY="sk-..."

Docs:
    https://openrouter.ai/docs
"""

from __future__ import annotations

import json
import logging
import os
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set to use OpenRouter backend."
            )

    # ------------------------------------------------------------------ #
    # Chat completion wrapper
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost/",
            "X-Title": "agi-hpc",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("[LLM][OpenRouter] request failed")
            raise

        out = resp.json()
        return out["choices"][0]["message"]["content"]
