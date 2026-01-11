# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LLM Adapter Protocol for AGI-HPC Left Hemisphere.

Defines the interface for LLM providers used in plan generation.
Supports multiple backends: Ollama (local), Anthropic (Claude), OpenAI (GPT).
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """
    Configuration for LLM adapters.

    Environment variables:
        AGI_LH_LLM_PROVIDER: ollama, anthropic, openai
        AGI_LH_LLM_MODEL: Model name/ID
        AGI_LH_LLM_API_KEY: API key (for cloud providers)
        AGI_LH_LLM_BASE_URL: Base URL (for Ollama or custom endpoints)
        AGI_LH_LLM_TEMPERATURE: Sampling temperature
        AGI_LH_LLM_MAX_TOKENS: Maximum tokens in response
    """

    provider: str = field(
        default_factory=lambda: os.getenv("AGI_LH_LLM_PROVIDER", "ollama")
    )
    model: str = field(
        default_factory=lambda: os.getenv("AGI_LH_LLM_MODEL", "llama3.2")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AGI_LH_LLM_API_KEY")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "AGI_LH_LLM_BASE_URL", "http://localhost:11434"
        )
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGI_LH_LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGI_LH_LLM_MAX_TOKENS", "2048"))
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("AGI_LH_LLM_TIMEOUT", "60.0"))
    )


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """
    Container for LLM response data.

    Attributes:
        content: The generated text content
        model: Model that generated the response
        usage: Token usage statistics (if available)
        raw: Raw response from the provider (for debugging)
    """

    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[Any] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get(
            "total_tokens", self.prompt_tokens + self.completion_tokens
        )


# ---------------------------------------------------------------------------
# LLM Adapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Protocol for LLM adapters.

    All LLM providers must implement this interface to be used
    with the LLM-powered planner.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text from the LLM.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse containing the generated content
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if the service is reachable and ready
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the name of the model being used."""
        ...


# ---------------------------------------------------------------------------
# Base Adapter Implementation
# ---------------------------------------------------------------------------


class BaseLLMAdapter(ABC):
    """
    Base class for LLM adapters with common functionality.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self._config = config or LLMConfig()
        logger.info(
            f"[LLM] Initializing {self.__class__.__name__} "
            f"model={self._config.model}"
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass

    def _get_temperature(self, override: Optional[float]) -> float:
        return override if override is not None else self._config.temperature

    def _get_max_tokens(self, override: Optional[int]) -> int:
        return override if override is not None else self._config.max_tokens


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_adapter(config: Optional[LLMConfig] = None) -> LLMAdapter:
    """
    Create an LLM adapter based on configuration.

    Args:
        config: LLM configuration (uses defaults if not provided)

    Returns:
        An LLMAdapter instance for the configured provider

    Raises:
        ValueError: If the provider is not supported
    """
    cfg = config or LLMConfig()

    if cfg.provider == "ollama":
        from agi.lh.llm.ollama import OllamaAdapter

        return OllamaAdapter(cfg)
    elif cfg.provider == "anthropic":
        from agi.lh.llm.anthropic import AnthropicAdapter

        return AnthropicAdapter(cfg)
    elif cfg.provider == "openai":
        from agi.lh.llm.openai import OpenAIAdapter

        return OpenAIAdapter(cfg)
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
