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

"""Unit tests for agi.meta.llm.config -- InferenceConfig and presets."""

from __future__ import annotations

from agi.meta.llm.config import InferenceConfig, LH_PRESET, RH_PRESET


class TestInferenceConfigDefaults:
    """Tests for InferenceConfig default values."""

    def test_default_temperature(self) -> None:
        cfg = InferenceConfig()
        assert cfg.temperature == 0.7

    def test_default_top_p(self) -> None:
        cfg = InferenceConfig()
        assert cfg.top_p == 0.95

    def test_default_max_tokens(self) -> None:
        cfg = InferenceConfig()
        assert cfg.max_tokens == 2048

    def test_default_empty_system_prompt(self) -> None:
        cfg = InferenceConfig()
        assert cfg.system_prompt == ""

    def test_default_empty_stop_sequences(self) -> None:
        cfg = InferenceConfig()
        assert cfg.stop_sequences == []

    def test_default_empty_model(self) -> None:
        cfg = InferenceConfig()
        assert cfg.model == ""

    def test_default_zero_penalties(self) -> None:
        cfg = InferenceConfig()
        assert cfg.presence_penalty == 0.0
        assert cfg.frequency_penalty == 0.0


class TestLHPreset:
    """Tests for the Left Hemisphere preset."""

    def test_lh_low_temperature(self) -> None:
        assert LH_PRESET.temperature == 0.3

    def test_lh_top_p(self) -> None:
        assert LH_PRESET.top_p == 0.90

    def test_lh_max_tokens(self) -> None:
        assert LH_PRESET.max_tokens == 4096

    def test_lh_system_prompt_contains_analytical(self) -> None:
        assert "analytical" in LH_PRESET.system_prompt.lower()

    def test_lh_system_prompt_contains_precise(self) -> None:
        assert "precise" in LH_PRESET.system_prompt.lower()


class TestRHPreset:
    """Tests for the Right Hemisphere preset."""

    def test_rh_high_temperature(self) -> None:
        assert RH_PRESET.temperature == 0.8

    def test_rh_top_p(self) -> None:
        assert RH_PRESET.top_p == 0.95

    def test_rh_max_tokens(self) -> None:
        assert RH_PRESET.max_tokens == 4096

    def test_rh_system_prompt_contains_creative(self) -> None:
        assert "creative" in RH_PRESET.system_prompt.lower()

    def test_rh_system_prompt_contains_pattern(self) -> None:
        assert "pattern" in RH_PRESET.system_prompt.lower()


class TestToApiParams:
    """Tests for InferenceConfig.to_api_params()."""

    def test_basic_params(self) -> None:
        cfg = InferenceConfig(temperature=0.5, top_p=0.9, max_tokens=1024)
        params = cfg.to_api_params()
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 1024

    def test_no_stop_when_empty(self) -> None:
        cfg = InferenceConfig()
        params = cfg.to_api_params()
        assert "stop" not in params

    def test_stop_sequences_included(self) -> None:
        cfg = InferenceConfig(stop_sequences=["\n", "END"])
        params = cfg.to_api_params()
        assert params["stop"] == ["\n", "END"]

    def test_model_included_when_set(self) -> None:
        cfg = InferenceConfig(model="gemma-4-27b")
        params = cfg.to_api_params()
        assert params["model"] == "gemma-4-27b"

    def test_model_excluded_when_empty(self) -> None:
        cfg = InferenceConfig()
        params = cfg.to_api_params()
        assert "model" not in params

    def test_penalties_included_when_nonzero(self) -> None:
        cfg = InferenceConfig(presence_penalty=0.5, frequency_penalty=0.3)
        params = cfg.to_api_params()
        assert params["presence_penalty"] == 0.5
        assert params["frequency_penalty"] == 0.3

    def test_penalties_excluded_when_zero(self) -> None:
        cfg = InferenceConfig()
        params = cfg.to_api_params()
        assert "presence_penalty" not in params
        assert "frequency_penalty" not in params

    def test_lh_preset_api_params(self) -> None:
        params = LH_PRESET.to_api_params()
        assert params["temperature"] == 0.3
        assert params["max_tokens"] == 4096
