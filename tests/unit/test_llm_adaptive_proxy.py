"""Tests for the adaptive LLM proxy mode selection logic."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestModeSelection:
    """Test the token estimation and mode switching logic."""

    def test_short_prompt_stays_fast(self) -> None:
        """Short prompts should use FAST mode."""
        from llm_adaptive_proxy import Mode, CHARS_PER_TOKEN

        # 100 tokens * 4 chars/token = 400 chars
        prompt_chars = 400
        est_tokens = prompt_chars // CHARS_PER_TOKEN
        threshold = 6000
        assert est_tokens < threshold
        target = Mode.LONG if est_tokens > threshold else Mode.FAST
        assert target == Mode.FAST

    def test_long_prompt_triggers_long(self) -> None:
        """Long prompts should trigger LONG mode."""
        from llm_adaptive_proxy import Mode, CHARS_PER_TOKEN

        # 8000 tokens * 4 chars/token = 32000 chars
        prompt_chars = 32000
        est_tokens = prompt_chars // CHARS_PER_TOKEN
        threshold = 6000
        assert est_tokens > threshold
        target = Mode.LONG if est_tokens > threshold else Mode.FAST
        assert target == Mode.LONG

    def test_token_estimation(self) -> None:
        """Token estimation should be conservative (4 chars/token)."""
        from llm_adaptive_proxy import CHARS_PER_TOKEN

        text = "Hello world, this is a test message for the proxy."
        est = len(text) // CHARS_PER_TOKEN
        # 50 chars / 4 = 12 tokens (real tokenizer would give ~11)
        assert 10 <= est <= 15

    def test_mode_configs_valid(self) -> None:
        """Both mode configs should have required fields."""
        from llm_adaptive_proxy import MODE_CONFIGS, Mode

        for mode in Mode:
            cfg = MODE_CONFIGS[mode]
            assert "ctx_size" in cfg
            assert "cache_k" in cfg
            assert "cache_v" in cfg
            assert "label" in cfg
            assert cfg["ctx_size"] > 0

    def test_fast_mode_is_fp16(self) -> None:
        from llm_adaptive_proxy import MODE_CONFIGS, Mode

        cfg = MODE_CONFIGS[Mode.FAST]
        assert cfg["cache_k"] == "f16"
        assert cfg["cache_v"] == "f16"
        assert cfg["ctx_size"] == 8192

    def test_long_mode_is_q8(self) -> None:
        from llm_adaptive_proxy import MODE_CONFIGS, Mode

        cfg = MODE_CONFIGS[Mode.LONG]
        assert cfg["cache_k"] == "q8_0"
        assert cfg["cache_v"] == "q8_0"
        assert cfg["ctx_size"] == 14336

    def test_long_context_is_bigger(self) -> None:
        from llm_adaptive_proxy import MODE_CONFIGS, Mode

        assert MODE_CONFIGS[Mode.LONG]["ctx_size"] > MODE_CONFIGS[Mode.FAST]["ctx_size"]
