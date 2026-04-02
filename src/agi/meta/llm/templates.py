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
Jinja2 prompt template registry for AGI-HPC cognitive services.

Manages system prompts for LH (analytical), RH (creative),
safety checks, and metacognitive reflection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from jinja2 import Environment, BaseLoader, TemplateSyntaxError
except ImportError:
    Environment = None  # type: ignore
    BaseLoader = None  # type: ignore
    TemplateSyntaxError = Exception  # type: ignore


# -----------------------------------------------------------------
# Built-in templates
# -----------------------------------------------------------------

_BUILTIN_TEMPLATES: Dict[str, str] = {
    "lh_analytical": (
        "You are the Left Hemisphere of Atlas, an AGI cognitive architecture "
        "running on dual Quadro GV100 GPUs. You are analytical, precise, and "
        "citation-heavy. You are powered by {{ model_name }}."
        "{% if rag_context %}\n\n"
        "--- Retrieved from local repositories ---\n"
        "{{ rag_context }}\n"
        "--- End of retrieved context ---\n"
        "{% endif %}"
        "\nWhen relevant context is provided, cite the repo and file. "
        "Reason step by step. Prefer accuracy over creativity."
    ),
    "rh_creative": (
        "You are the Right Hemisphere of Atlas, an AGI cognitive architecture "
        "running on dual Quadro GV100 GPUs. You are creative, pattern-seeking, "
        "and intuitive. You are powered by {{ model_name }}."
        "{% if rag_context %}\n\n"
        "--- Retrieved from local repositories ---\n"
        "{{ rag_context }}\n"
        "--- End of retrieved context ---\n"
        "{% endif %}"
        "\nThink in analogies, metaphors, and cross-cutting connections. "
        "Generate diverse possibilities before converging."
    ),
    "safety_check": (
        "You are the Safety subsystem of Atlas AGI. Evaluate the following "
        "{{ check_type }} for potential risks:\n\n"
        "Content: {{ content }}\n\n"
        "Respond with a JSON object: "
        '{"safe": true/false, "risk_level": "low/medium/high", '
        '"reason": "explanation"}'
    ),
    "metacognition_reflect": (
        "You are the Metacognition module of Atlas AGI. Reflect on the "
        "following chain of thought from the {{ hemisphere }} hemisphere:\n\n"
        "Query: {{ query }}\n"
        "Response: {{ response }}\n"
        "Latency: {{ latency_ms }}ms\n"
        "Tokens: {{ tokens_used }}\n\n"
        "Evaluate: quality, confidence, completeness, and whether the "
        "response should be revised or sent to the other hemisphere."
    ),
}


class PromptTemplateRegistry:
    """Registry for Jinja2 prompt templates.

    Manages built-in and custom templates for cognitive services.

    Usage::

        registry = PromptTemplateRegistry()
        prompt = registry.render("lh_analytical",
                                  model_name="Gemma 4",
                                  rag_context="...")
    """

    def __init__(self) -> None:
        if Environment is None:
            raise RuntimeError(
                "jinja2 is required but not installed. "
                "Install with: pip install jinja2"
            )
        self._env = Environment(loader=BaseLoader())
        self._templates: Dict[str, str] = dict(_BUILTIN_TEMPLATES)

    def register(self, name: str, template_str: str) -> None:
        """Register a custom template.

        Args:
            name: Template identifier.
            template_str: Jinja2 template string.
        """
        # Validate syntax eagerly
        self._env.parse(template_str)
        self._templates[name] = template_str
        logger.info("[templates] registered template '%s'", name)

    def render(self, name: str, **kwargs: Any) -> str:
        """Render a template by name.

        Args:
            name: Template identifier.
            **kwargs: Variables to inject into the template.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If the template name is not registered.
        """
        if name not in self._templates:
            raise KeyError(
                f"Template '{name}' not found. "
                f"Available: {list(self._templates.keys())}"
            )
        tmpl = self._env.from_string(self._templates[name])
        return tmpl.render(**kwargs)

    def list_templates(self) -> list[str]:
        """Return names of all registered templates."""
        return list(self._templates.keys())
