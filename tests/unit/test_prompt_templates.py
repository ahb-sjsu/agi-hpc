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

"""Unit tests for agi.meta.llm.templates -- PromptTemplateRegistry."""

from __future__ import annotations

import pytest

from agi.meta.llm.templates import PromptTemplateRegistry


class TestPromptTemplateRegistry:
    """Tests for PromptTemplateRegistry."""

    def test_init_loads_builtins(self) -> None:
        reg = PromptTemplateRegistry()
        templates = reg.list_templates()
        assert "lh_analytical" in templates
        assert "rh_creative" in templates
        assert "safety_check" in templates
        assert "metacognition_reflect" in templates

    def test_render_lh_analytical(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render("lh_analytical", model_name="Gemma 4", rag_context="")
        assert "Left Hemisphere" in rendered
        assert "Gemma 4" in rendered
        assert "analytical" in rendered.lower()

    def test_render_lh_analytical_with_rag(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render(
            "lh_analytical",
            model_name="Gemma 4",
            rag_context="File: utils.py\nContent: def hello(): pass",
        )
        assert "Retrieved from local repositories" in rendered
        assert "utils.py" in rendered

    def test_render_lh_analytical_without_rag(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render("lh_analytical", model_name="Gemma 4", rag_context="")
        assert "Retrieved from local repositories" not in rendered

    def test_render_rh_creative(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render("rh_creative", model_name="Mistral", rag_context="")
        assert "Right Hemisphere" in rendered
        assert "Mistral" in rendered
        assert "creative" in rendered.lower()

    def test_render_safety_check(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render(
            "safety_check",
            check_type="input",
            content="Hello world",
        )
        assert "Safety subsystem" in rendered
        assert "input" in rendered
        assert "Hello world" in rendered

    def test_render_metacognition_reflect(self) -> None:
        reg = PromptTemplateRegistry()
        rendered = reg.render(
            "metacognition_reflect",
            hemisphere="lh",
            query="What is AI?",
            response="AI is...",
            latency_ms=42,
            tokens_used=100,
        )
        assert "Metacognition" in rendered
        assert "lh" in rendered
        assert "What is AI?" in rendered

    def test_missing_template_raises_key_error(self) -> None:
        reg = PromptTemplateRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            reg.render("nonexistent")

    def test_register_custom_template(self) -> None:
        reg = PromptTemplateRegistry()
        reg.register("custom", "Hello {{ user_name }}!")
        result = reg.render("custom", user_name="Atlas")
        assert result == "Hello Atlas!"

    def test_register_appears_in_list(self) -> None:
        reg = PromptTemplateRegistry()
        reg.register("my_tmpl", "test {{ x }}")
        assert "my_tmpl" in reg.list_templates()

    def test_builtin_template_count(self) -> None:
        reg = PromptTemplateRegistry()
        assert len(reg.list_templates()) == 4
