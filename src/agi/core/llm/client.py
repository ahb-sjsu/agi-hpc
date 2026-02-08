# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Main LLM Client with middleware stack support.

Provides a unified interface for LLM operations with:
- Provider abstraction (Anthropic, OpenAI, Ollama)
- Middleware stack (logging, retry, rate limiting)
- Sync and async support
- Streaming support

Usage:
    from agi.core.llm import LLMClient

    client = LLMClient()  # Uses env config
    response = client.complete("What is the capital of France?")

    # With specific provider
    client = LLMClient(provider="anthropic", model="claude-3-5-sonnet-20241022")

    # Streaming
    for chunk in client.stream("Tell me a story"):
        print(chunk.delta, end="")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Type, Union

from agi.core.llm.config import LLMConfig, get_default_model
from agi.core.llm.providers.base import BaseProvider, LLMProvider
from agi.core.llm.types import (
    CompletionRequest,
    LLMResponse,
    Message,
    StreamChunk,
    StreamingResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware Types
# ---------------------------------------------------------------------------

Middleware = Callable[[CompletionRequest, Callable], LLMResponse]


@dataclass
class RetryConfig:
    """Configuration for retry middleware."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)


class RetryMiddleware:
    """Middleware that retries failed requests with exponential backoff."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        last_exception = None
        delay = self.config.base_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                return next_handler(request)
            except self.config.retryable_exceptions as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    logger.warning(
                        "[llm][retry] attempt %d failed: %s, retrying in %.1fs",
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(
                        delay * self.config.exponential_base,
                        self.config.max_delay,
                    )

        raise last_exception or RuntimeError("Retry failed")


class LoggingMiddleware:
    """Middleware that logs requests and responses."""

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        model = request.model or "default"
        msg_count = len(request.messages)
        logger.info("[llm][request] model=%s messages=%d", model, msg_count)

        start = time.perf_counter()
        response = next_handler(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "[llm][response] model=%s tokens=%d latency=%.1fms",
            response.model,
            response.total_tokens,
            elapsed,
        )

        return response


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Unified LLM client with provider abstraction and middleware support.

    Provides a high-level interface for LLM operations that can be used
    across all AGI-HPC subsystems (LH, RH, Memory, Safety, etc.).
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        middleware: Optional[List[Middleware]] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration (uses defaults if not provided)
            provider: Override provider from config
            model: Override model from config
            middleware: Custom middleware stack
        """
        self._config = config or LLMConfig()

        if provider:
            self._config = self._config.with_provider(provider)
        if model:
            self._config = self._config.with_model(model)

        self._provider = self._create_provider()
        self._middleware = middleware or self._default_middleware()

        logger.info(
            "[llm] client initialized provider=%s model=%s",
            self._config.provider,
            self._provider.model,
        )

    def _create_provider(self) -> LLMProvider:
        """Create the LLM provider based on configuration."""
        provider_name = self._config.provider.lower()

        if provider_name == "anthropic":
            from agi.core.llm.providers.anthropic import AnthropicProvider
            return AnthropicProvider(self._config)
        elif provider_name == "openai":
            from agi.core.llm.providers.openai import OpenAIProvider
            return OpenAIProvider(self._config)
        elif provider_name == "ollama":
            from agi.core.llm.providers.ollama import OllamaProvider
            return OllamaProvider(self._config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def _default_middleware(self) -> List[Middleware]:
        """Create default middleware stack."""
        middleware = []

        # Add logging if debug enabled
        if logger.isEnabledFor(logging.DEBUG):
            middleware.append(LoggingMiddleware())

        # Add retry middleware
        if self._config.max_retries > 0:
            middleware.append(RetryMiddleware(RetryConfig(
                max_retries=self._config.max_retries,
                base_delay=self._config.retry_delay,
            )))

        return middleware

    def _execute_with_middleware(
        self,
        request: CompletionRequest,
    ) -> LLMResponse:
        """Execute request through middleware stack."""

        def final_handler(req: CompletionRequest) -> LLMResponse:
            return self._provider.complete(req)

        # Build middleware chain from inside out
        handler = final_handler
        for mw in reversed(self._middleware):
            prev_handler = handler

            def make_next(mw, prev):
                return lambda req: mw(req, prev)

            handler = make_next(mw, prev_handler)

        return handler(request)

    @property
    def provider(self) -> LLMProvider:
        """Get the underlying provider."""
        return self._provider

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._provider.model

    def complete(
        self,
        prompt: Union[str, List[Message], CompletionRequest],
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute a completion request.

        Args:
            prompt: User prompt string, list of messages, or CompletionRequest
            system: Optional system prompt (when prompt is a string)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content
        """
        if isinstance(prompt, CompletionRequest):
            request = prompt
        elif isinstance(prompt, list):
            request = CompletionRequest(messages=prompt, **kwargs)
        else:
            request = CompletionRequest.simple(prompt, system=system, **kwargs)

        return self._execute_with_middleware(request)

    def stream(
        self,
        prompt: Union[str, List[Message], CompletionRequest],
        system: Optional[str] = None,
        **kwargs,
    ) -> StreamingResponse:
        """
        Execute a streaming completion request.

        Args:
            prompt: User prompt string, list of messages, or CompletionRequest
            system: Optional system prompt (when prompt is a string)
            **kwargs: Additional parameters

        Returns:
            StreamingResponse iterator
        """
        if isinstance(prompt, CompletionRequest):
            request = prompt
        elif isinstance(prompt, list):
            request = CompletionRequest(messages=prompt, stream=True, **kwargs)
        else:
            request = CompletionRequest.simple(
                prompt, system=system, stream=True, **kwargs
            )

        return self._provider.stream(request)

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self._provider.is_available()


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Create an LLM client with the specified configuration.

    Args:
        provider: Provider name (anthropic, openai, ollama)
        model: Model name
        **kwargs: Additional configuration options

    Returns:
        Configured LLMClient instance
    """
    config = LLMConfig(**kwargs)
    return LLMClient(config=config, provider=provider, model=model)


def get_provider(
    name: str,
    config: Optional[LLMConfig] = None,
) -> LLMProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name
        config: Optional configuration

    Returns:
        Provider instance
    """
    config = config or LLMConfig()

    if name == "anthropic":
        from agi.core.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif name == "openai":
        from agi.core.llm.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    elif name == "ollama":
        from agi.core.llm.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider: {name}")
