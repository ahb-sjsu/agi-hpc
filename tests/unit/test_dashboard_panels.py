# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for dashboard panel elements.

Validates that the HTML dashboard contains all required panels,
element IDs, and JavaScript bindings for safety, dreaming,
training, and privilege subsystems.
"""

from __future__ import annotations

from pathlib import Path

DASHBOARD = Path("atlas-chat-schematic.html")


class TestDashboardExists:
    """Basic dashboard file tests."""

    def test_file_exists(self) -> None:
        assert DASHBOARD.exists()

    def test_is_html(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content


class TestSafetyPanel:
    """Tests for the Safety Gateway dashboard panel."""

    def test_safety_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-safety"' in content

    def test_safety_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="safety-in"' in content
        assert 'id="safety-out"' in content
        assert 'id="safety-vetoes"' in content
        assert 'id="safety-latency"' in content
        assert 'id="safety-audit"' in content
        assert 'id="safety-layer"' in content

    def test_safety_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "safety.input_checks" in content
        assert "safety.avg_latency_ms" in content


class TestDreamingPanel:
    """Tests for the Dreaming (DMN) dashboard panel."""

    def test_dreaming_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-dreaming"' in content

    def test_dreaming_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="dream-episodes"' in content
        assert 'id="dream-unconsolidated"' in content
        assert 'id="dream-articles"' in content

    def test_dreaming_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "episodic_episodes" in content


class TestTrainingPanel:
    """Tests for the DM Training dashboard panel."""

    def test_training_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-training"' in content

    def test_training_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="training-sessions"' in content
        assert 'id="training-score"' in content

    def test_training_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "total_sessions" in content
        assert "last_session_score" in content


class TestPrivilegePanel:
    """Tests for the Ego Privileges dashboard panel."""

    def test_privilege_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-privileges"' in content

    def test_privilege_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="priv-level"' in content
        assert 'id="priv-name"' in content
        assert 'id="priv-score"' in content
        assert 'id="priv-vetoes"' in content
        assert 'id="priv-episodes"' in content

    def test_privilege_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "ego_privileges" in content
        assert "level_name" in content
        assert "mean_score" in content

    def test_privilege_color_coding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "green-dim" in content  # L1-L2 color
        assert "yellow-dim" in content  # L3+ color


class TestFreudianLabels:
    """Tests that dashboard uses Freudian psyche terminology."""

    def test_superego_label(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "Superego" in content

    def test_id_label(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert ">Id<" in content or "Id /" in content

    def test_ego_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-ego"' in content

    def test_ego_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="ego-model"' in content
        assert 'id="ego-role"' in content
        assert 'id="ego-slots"' in content

    def test_ego_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "ego.status" in content
        assert "updatePsycheCard" in content
        assert "'sc-ego'" in content

    def test_gpu_labels_freudian(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "GPU 0 (Superego)" in content
        assert "GPU 1 (Id)" in content

    def test_bottom_bar_ego(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "CPU (Ego)" in content


class TestAttentionPanel:
    """Tests for the Attention Filter dashboard panel."""

    def test_attention_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-attention"' in content

    def test_attention_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="attn-detected"' in content
        assert 'id="attn-warnings"' in content
        assert 'id="attn-intensity"' in content
        assert 'id="attn-score"' in content

    def test_attention_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "t.attention" in content
        assert "last_intensity" in content

    def test_attention_in_topology(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "Attention Filter" in content


class TestCurriculumPanel:
    """Tests for the Curriculum (gap detection) dashboard panel."""

    def test_curriculum_card_exists(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="sc-curriculum"' in content

    def test_curriculum_elements(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="curriculum-gaps"' in content
        assert 'id="curriculum-focus"' in content

    def test_curriculum_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "t.curriculum" in content
        assert "gaps_detected" in content


class TestDreamingEnriched:
    """Tests that dreaming panel shows insights and grades."""

    def test_dream_insights_element(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="dream-insights"' in content

    def test_dream_grade_element(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="dream-grade"' in content

    def test_dream_js_uses_dreaming_data(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "t.dreaming" in content


class TestTrainingEnriched:
    """Tests that training panel shows retrospective count."""

    def test_retro_element(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert 'id="training-retro"' in content

    def test_retro_js_binding(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "retrospective_used" in content


class TestCogSciLabels:
    """Tests that dashboard uses cognitive science terminology."""

    def test_somatic_marker(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "Somatic Marker" in content

    def test_default_mode_network(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "DMN" in content

    def test_kohlberg(self) -> None:
        content = DASHBOARD.read_text(encoding="utf-8")
        assert "Kohlberg" in content
