# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Base provider interface for LLM backends.

Defines the protocol that all LLM providers must implement.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional, Protocol, runtime_checkable

from agi.core.llm.config import LLMConfig
from agi.core.llm.types import (
    CompletionRequest,
    LLMResponse,
    StreamChunk,
    StreamingResponse,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def model(self) -> str:
        """Current model name."""
        ...

    def complete(self, request: CompletionRequest) -> LLMResponse:
        """Execute a completion request."""
        ...

    def stream(self, request: CompletionRequest) -> StreamingResponse:
        """Execute a streaming completion request."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is available."""
        ...


class BaseProvider(ABC):
    """
    Base class for LLM providers.

    Provides common functionality like logging, timing, and error handling.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self._config = config or LLMConfig()
        self._model = self._config.model or self._get_default_model()
        logger.info(
            "[llm][%s] initialized model=%s",
            self.name,
            self._model,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    def model(self) -> str:
        """Current model name."""
        return self._model

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    def _complete_impl(self, request: CompletionRequest) -> LLMResponse:
        """Implementation of completion request."""
        pass

    @abstractmethod
    def _stream_impl(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Implementation of streaming completion request."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    def complete(self, request: CompletionRequest) -> LLMResponse:
        """Execute a completion request with timing and error handling."""
        start = time.perf_counter()
        try:
            response = self._complete_impl(request)
            response.latency_ms = (time.perf_counter() - start) * 1000
            response.provider = self.name
            if not response.model:
                response.model = self._model
            logger.debug(
                "[llm][%s] completed tokens=%d latency=%.1fms",
                self.name,
                response.total_tokens,
                response.latency_ms,
            )
            return response
        except Exception as e:
            logger.exception("[llm][%s] completion failed", self.name)
            raise

    def stream(self, request: CompletionRequest) -> StreamingResponse:
        """Execute a streaming completion request."""
        try:
            iterator = self._stream_impl(request)
            return StreamingResponse(
                iterator=iterator,
                model=self._model,
                provider=self.name,
            )
        except Exception as e:
            logger.exception("[llm][%s] stream failed", self.name)
            raise

    def _get_temperature(self, request: CompletionRequest) -> float:
        """Get temperature from request or config."""
        if request.temperature is not None:
            return request.temperature
        return self._config.temperature

    def _get_max_tokens(self, request: CompletionRequest) -> int:
        """Get max tokens from request or config."""
        if request.max_tokens is not None:
            return request.max_tokens
        return self._config.max_tokens

    def _get_model(self, request: CompletionRequest) -> str:
        """Get model from request or config."""
        if request.model:
            return request.model
        return self._model


class AsyncLLMProvider(Protocol):
    """Protocol for async LLM providers."""

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def model(self) -> str:
        """Current model name."""
        ...

    async def complete(self, request: CompletionRequest) -> LLMResponse:
        """Execute an async completion request."""
        ...

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Execute an async streaming completion request."""
        ...

    async def is_available(self) -> bool:
        """Check if the provider is available."""
        ...


class BaseAsyncProvider(ABC):
    """
    Base class for async LLM providers.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self._config = config or LLMConfig()
        self._model = self._config.model or self._get_default_model()
        logger.info(
            "[llm][%s][async] initialized model=%s",
            self.name,
            self._model,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    def model(self) -> str:
        """Current model name."""
        return self._model

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    async def _complete_impl(self, request: CompletionRequest) -> LLMResponse:
        """Implementation of completion request."""
        pass

    @abstractmethod
    async def _stream_impl(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Implementation of streaming completion request."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    async def complete(self, request: CompletionRequest) -> LLMResponse:
        """Execute a completion request with timing and error handling."""
        start = time.perf_counter()
        try:
            response = await self._complete_impl(request)
            response.latency_ms = (time.perf_counter() - start) * 1000
            response.provider = self.name
            if not response.model:
                response.model = self._model
            logger.debug(
                "[llm][%s][async] completed tokens=%d latency=%.1fms",
                self.name,
                response.total_tokens,
                response.latency_ms,
            )
            return response
        except Exception as e:
            logger.exception("[llm][%s][async] completion failed", self.name)
            raise

    def _get_temperature(self, request: CompletionRequest) -> float:
        """Get temperature from request or config."""
        if request.temperature is not None:
            return request.temperature
        return self._config.temperature

    def _get_max_tokens(self, request: CompletionRequest) -> int:
        """Get max tokens from request or config."""
        if request.max_tokens is not None:
            return request.max_tokens
        return self._config.max_tokens

    def _get_model(self, request: CompletionRequest) -> str:
        """Get model from request or config."""
        if request.model:
            return request.model
        return self._model
