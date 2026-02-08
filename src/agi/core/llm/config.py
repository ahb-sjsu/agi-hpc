# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Configuration management for LLM infrastructure.

Supports environment variables and programmatic configuration:
    AGI_LLM_PROVIDER        Provider name (anthropic, openai, ollama, openrouter)
    AGI_LLM_MODEL           Model identifier
    AGI_LLM_API_KEY         API key for cloud providers
    AGI_LLM_BASE_URL        Base URL for API (optional)
    AGI_LLM_TEMPERATURE     Default temperature (0.0-2.0)
    AGI_LLM_MAX_TOKENS      Default max tokens
    AGI_LLM_TIMEOUT         Request timeout in seconds
    AGI_LLM_MAX_RETRIES     Maximum retry attempts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """
    Global LLM configuration.

    Can be initialized from environment or programmatically.
    """

    # Provider selection
    provider: str = field(
        default_factory=lambda: os.getenv("AGI_LLM_PROVIDER", "ollama")
    )

    # Model configuration
    model: str = field(
        default_factory=lambda: os.getenv("AGI_LLM_MODEL", "")
    )

    # Authentication
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AGI_LLM_API_KEY")
    )

    # Endpoint configuration
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("AGI_LLM_BASE_URL")
    )

    # Generation defaults
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGI_LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGI_LLM_MAX_TOKENS", "2048"))
    )
    top_p: float = field(
        default_factory=lambda: float(os.getenv("AGI_LLM_TOP_P", "1.0"))
    )

    # Request configuration
    timeout: float = field(
        default_factory=lambda: float(os.getenv("AGI_LLM_TIMEOUT", "60.0"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("AGI_LLM_MAX_RETRIES", "3"))
    )
    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("AGI_LLM_RETRY_DELAY", "1.0"))
    )

    # Rate limiting
    rate_limit_rpm: Optional[int] = field(
        default_factory=lambda: int(os.getenv("AGI_LLM_RATE_LIMIT_RPM", "0")) or None
    )
    rate_limit_tpm: Optional[int] = field(
        default_factory=lambda: int(os.getenv("AGI_LLM_RATE_LIMIT_TPM", "0")) or None
    )

    # Provider-specific configs
    provider_configs: Dict[str, ProviderConfig] = field(default_factory=dict)

    def get_provider_config(self, provider: Optional[str] = None) -> ProviderConfig:
        """Get configuration for a specific provider."""
        provider = provider or self.provider
        if provider in self.provider_configs:
            return self.provider_configs[provider]
        return ProviderConfig(
            name=provider,
            api_key=self.api_key,
            base_url=self.base_url,
            default_model=self.model,
        )

    def with_model(self, model: str) -> "LLMConfig":
        """Create a copy with a different model."""
        return LLMConfig(
            provider=self.provider,
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            rate_limit_rpm=self.rate_limit_rpm,
            rate_limit_tpm=self.rate_limit_tpm,
            provider_configs=self.provider_configs,
        )

    def with_provider(self, provider: str) -> "LLMConfig":
        """Create a copy with a different provider."""
        return LLMConfig(
            provider=provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            rate_limit_rpm=self.rate_limit_rpm,
            rate_limit_tpm=self.rate_limit_tpm,
            provider_configs=self.provider_configs,
        )


# Default provider model mappings
DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "ollama": "llama3.2",
    "openrouter": "anthropic/claude-3.5-sonnet",
}


def get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    return DEFAULT_MODELS.get(provider, "")
