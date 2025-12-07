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
