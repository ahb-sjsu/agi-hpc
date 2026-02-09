# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LLM Middleware Stack for AGI-HPC.

Provides composable middleware for LLM operations:
- CachingMiddleware: Memory/Redis caching with TTL
- RateLimitMiddleware: Token bucket rate limiting
- RetryMiddleware: Exponential backoff with jitter
- FallbackMiddleware: Provider fallback chain
- LoggingMiddleware: Request/response logging

Usage:
    from agi.core.llm.middleware import (
        CachingMiddleware,
        RateLimitMiddleware,
        RetryMiddleware,
    )

    client = LLMClient(middleware=[
        CachingMiddleware(ttl=300),
        RateLimitMiddleware(requests_per_minute=60),
        RetryMiddleware(max_retries=3),
    ])
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from agi.core.llm.types import CompletionRequest, LLMResponse, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware Protocol
# ---------------------------------------------------------------------------


class Middleware(Protocol):
    """Protocol for LLM middleware."""

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        """Process request through middleware."""
        ...


# ---------------------------------------------------------------------------
# Caching Middleware
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Get cached value."""
        pass

    @abstractmethod
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with TTL support."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[bytes, float]] = {}
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key not in self._cache:
                return None

            value, expires_at = self._cache[key]
            if expires_at > 0 and time.time() > expires_at:
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            expires_at = time.time() + ttl if ttl else 0
            self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "llm:cache:",
    ):
        try:
            import redis

            self._redis = redis.from_url(url)
        except ImportError:
            raise RuntimeError("redis-py required for RedisCacheBackend") from None

        self._prefix = prefix

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[bytes]:
        return self._redis.get(self._make_key(key))

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        redis_key = self._make_key(key)
        if ttl:
            self._redis.setex(redis_key, ttl, value)
        else:
            self._redis.set(redis_key, value)

    def delete(self, key: str) -> bool:
        return self._redis.delete(self._make_key(key)) > 0


@dataclass
class CacheConfig:
    """Configuration for caching middleware."""

    enabled: bool = True
    ttl: int = 300  # seconds
    max_size: int = 1000
    backend: str = "memory"  # memory, redis
    redis_url: str = "redis://localhost:6379"
    include_system_prompt: bool = True
    include_temperature: bool = False


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CachingMiddleware:
    """
    Middleware that caches LLM responses.

    Generates cache keys from request content and caches
    successful responses with configurable TTL.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self._config = config or CacheConfig()
        self._stats = CacheStats()
        self._backend = self._create_backend()

        logger.info(
            "[llm][cache] initialized backend=%s ttl=%d",
            self._config.backend,
            self._config.ttl,
        )

    def _create_backend(self) -> CacheBackend:
        if self._config.backend == "redis":
            return RedisCacheBackend(url=self._config.redis_url)
        return MemoryCacheBackend(max_size=self._config.max_size)

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        if not self._config.enabled:
            return next_handler(request)

        # Generate cache key
        cache_key = self._generate_key(request)

        # Try cache
        cached = self._backend.get(cache_key)
        if cached:
            self._stats.hits += 1
            logger.debug("[llm][cache] hit key=%s", cache_key[:16])
            return self._deserialize_response(cached)

        self._stats.misses += 1

        # Execute request
        response = next_handler(request)

        # Cache successful response
        if response.content:
            serialized = self._serialize_response(response)
            self._backend.set(cache_key, serialized, self._config.ttl)
            self._stats.sets += 1
            logger.debug("[llm][cache] set key=%s", cache_key[:16])

        return response

    def _generate_key(self, request: CompletionRequest) -> str:
        """Generate cache key from request."""
        parts = []

        # Add messages
        for msg in request.messages:
            parts.append(f"{msg.role}:{msg.content}")

        # Optionally include system prompt
        if self._config.include_system_prompt and request.system:
            parts.append(f"system:{request.system}")

        # Optionally include temperature
        if self._config.include_temperature:
            parts.append(f"temp:{request.temperature}")

        # Add model
        if request.model:
            parts.append(f"model:{request.model}")

        key_data = "|".join(parts)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _serialize_response(self, response: LLMResponse) -> bytes:
        """Serialize response for caching."""
        data = {
            "content": response.content,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        }
        return json.dumps(data).encode()

    def _deserialize_response(self, data: bytes) -> LLMResponse:
        """Deserialize cached response."""
        parsed = json.loads(data.decode())
        return LLMResponse(
            content=parsed["content"],
            model=parsed["model"],
            prompt_tokens=parsed.get("prompt_tokens", 0),
            completion_tokens=parsed.get("completion_tokens", 0),
            total_tokens=parsed.get("total_tokens", 0),
        )

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def clear(self) -> None:
        """Clear the cache."""
        if isinstance(self._backend, MemoryCacheBackend):
            self._backend.clear()


# ---------------------------------------------------------------------------
# Rate Limiting Middleware
# ---------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    burst_multiplier: float = 1.5
    wait_on_limit: bool = True
    max_wait_seconds: float = 60.0


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(
        self,
        rate: float,
        capacity: float,
    ):
        self._rate = rate  # tokens per second
        self._capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(
        self, tokens: int = 1, wait: bool = True, max_wait: float = 60.0
    ) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            wait: Whether to wait for tokens
            max_wait: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not wait:
                return False

            # Calculate wait time
            needed = tokens - self._tokens
            wait_time = needed / self._rate

            if wait_time > max_wait:
                return False

        # Wait outside lock
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate,
        )
        self._last_update = now

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class RateLimitMiddleware:
    """
    Middleware that enforces rate limits.

    Uses token bucket algorithm for both request rate
    and token consumption limiting.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self._config = config or RateLimitConfig()

        # Request rate bucket
        request_rate = self._config.requests_per_minute / 60.0
        request_capacity = request_rate * self._config.burst_multiplier * 60
        self._request_bucket = TokenBucket(request_rate, request_capacity)

        # Token rate bucket
        token_rate = self._config.tokens_per_minute / 60.0
        token_capacity = token_rate * self._config.burst_multiplier * 60
        self._token_bucket = TokenBucket(token_rate, token_capacity)

        self._total_requests = 0
        self._total_tokens = 0
        self._rate_limited_count = 0

        logger.info(
            "[llm][ratelimit] initialized rpm=%d tpm=%d",
            self._config.requests_per_minute,
            self._config.tokens_per_minute,
        )

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        # Acquire request token
        acquired = self._request_bucket.acquire(
            tokens=1,
            wait=self._config.wait_on_limit,
            max_wait=self._config.max_wait_seconds,
        )

        if not acquired:
            self._rate_limited_count += 1
            raise RateLimitExceeded("Request rate limit exceeded")

        self._total_requests += 1

        # Execute request
        response = next_handler(request)

        # Consume token budget
        if response.total_tokens > 0:
            self._token_bucket.acquire(
                tokens=response.total_tokens,
                wait=False,  # Don't block after response
            )
            self._total_tokens += response.total_tokens

        return response

    def get_wait_time(self) -> float:
        """Get estimated wait time until next request allowed."""
        available = self._request_bucket.available
        if available >= 1:
            return 0.0
        return (1 - available) / (self._config.requests_per_minute / 60.0)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limited_count": self._rate_limited_count,
            "available_request_tokens": self._request_bucket.available,
            "available_token_budget": self._token_bucket.available,
        }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


# ---------------------------------------------------------------------------
# Retry Middleware
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for retry middleware."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)
    retryable_exceptions: Tuple[type, ...] = (
        ConnectionError,
        TimeoutError,
        RateLimitExceeded,
    )


class RetryMiddleware:
    """
    Middleware that retries failed requests with exponential backoff.

    Implements jitter to avoid thundering herd problem.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self._config = config or RetryConfig()
        self._total_retries = 0

        logger.info(
            "[llm][retry] initialized max_retries=%d base_delay=%.1f",
            self._config.max_retries,
            self._config.base_delay,
        )

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        last_exception = None
        delay = self._config.base_delay

        for attempt in range(self._config.max_retries + 1):
            try:
                return next_handler(request)
            except self._config.retryable_exceptions as e:
                last_exception = e

                if attempt < self._config.max_retries:
                    # Add jitter
                    jitter = random.uniform(
                        -self._config.jitter * delay,
                        self._config.jitter * delay,
                    )
                    sleep_time = delay + jitter

                    logger.warning(
                        "[llm][retry] attempt %d failed: %s, retrying in %.2fs",
                        attempt + 1,
                        type(e).__name__,
                        sleep_time,
                    )

                    time.sleep(sleep_time)
                    self._total_retries += 1

                    # Exponential backoff
                    delay = min(
                        delay * self._config.exponential_base,
                        self._config.max_delay,
                    )

        logger.error(
            "[llm][retry] all %d attempts failed",
            self._config.max_retries + 1,
        )
        raise last_exception or RuntimeError("Retry failed")

    @property
    def total_retries(self) -> int:
        return self._total_retries


# ---------------------------------------------------------------------------
# Fallback Middleware
# ---------------------------------------------------------------------------


@dataclass
class FallbackConfig:
    """Configuration for provider fallback."""

    providers: List[str] = field(default_factory=lambda: ["anthropic", "openai"])
    fail_fast: bool = False


class FallbackMiddleware:
    """
    Middleware that falls back to alternative providers on failure.

    Tries each provider in order until one succeeds.
    """

    def __init__(
        self,
        provider_factory: Callable[[str], Callable[[CompletionRequest], LLMResponse]],
        config: Optional[FallbackConfig] = None,
    ):
        self._provider_factory = provider_factory
        self._config = config or FallbackConfig()
        self._fallback_count = 0

        logger.info(
            "[llm][fallback] initialized providers=%s",
            self._config.providers,
        )

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        last_exception = None

        # Try primary handler first
        try:
            return next_handler(request)
        except Exception as e:
            last_exception = e
            logger.warning(
                "[llm][fallback] primary failed: %s, trying fallbacks",
                e,
            )

        if self._config.fail_fast:
            raise last_exception

        # Try fallback providers
        for provider_name in self._config.providers[1:]:
            try:
                fallback_handler = self._provider_factory(provider_name)
                response = fallback_handler(request)
                self._fallback_count += 1
                logger.info(
                    "[llm][fallback] succeeded with provider=%s",
                    provider_name,
                )
                return response
            except Exception as e:
                last_exception = e
                logger.warning(
                    "[llm][fallback] provider %s failed: %s",
                    provider_name,
                    e,
                )

        raise last_exception or RuntimeError("All providers failed")

    @property
    def fallback_count(self) -> int:
        return self._fallback_count


# ---------------------------------------------------------------------------
# Logging Middleware
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Configuration for logging middleware."""

    log_requests: bool = True
    log_responses: bool = True
    log_content: bool = False  # Log actual content (may be sensitive)
    max_content_length: int = 100


class LoggingMiddleware:
    """Middleware that logs requests and responses."""

    def __init__(self, config: Optional[LoggingConfig] = None):
        self._config = config or LoggingConfig()

    def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], LLMResponse],
    ) -> LLMResponse:
        start_time = time.perf_counter()

        if self._config.log_requests:
            self._log_request(request)

        response = next_handler(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if self._config.log_responses:
            self._log_response(response, elapsed_ms)

        return response

    def _log_request(self, request: CompletionRequest) -> None:
        msg_count = len(request.messages)
        model = request.model or "default"

        if self._config.log_content and request.messages:
            last_msg = request.messages[-1].content
            if len(last_msg) > self._config.max_content_length:
                last_msg = last_msg[: self._config.max_content_length] + "..."
            logger.info(
                "[llm][request] model=%s messages=%d content=%s",
                model,
                msg_count,
                last_msg,
            )
        else:
            logger.info(
                "[llm][request] model=%s messages=%d",
                model,
                msg_count,
            )

    def _log_response(self, response: LLMResponse, elapsed_ms: float) -> None:
        if self._config.log_content and response.content:
            content = response.content
            if len(content) > self._config.max_content_length:
                content = content[: self._config.max_content_length] + "..."
            logger.info(
                "[llm][response] model=%s tokens=%d latency=%.1fms content=%s",
                response.model,
                response.total_tokens,
                elapsed_ms,
                content,
            )
        else:
            logger.info(
                "[llm][response] model=%s tokens=%d latency=%.1fms",
                response.model,
                response.total_tokens,
                elapsed_ms,
            )


# ---------------------------------------------------------------------------
# Middleware Builder
# ---------------------------------------------------------------------------


class MiddlewareStack:
    """Builder for creating middleware stacks."""

    def __init__(self):
        self._middleware: List[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareStack":
        """Add middleware to the stack."""
        self._middleware.append(middleware)
        return self

    def with_caching(
        self,
        ttl: int = 300,
        backend: str = "memory",
        **kwargs,
    ) -> "MiddlewareStack":
        """Add caching middleware."""
        config = CacheConfig(ttl=ttl, backend=backend, **kwargs)
        return self.add(CachingMiddleware(config))

    def with_rate_limiting(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        **kwargs,
    ) -> "MiddlewareStack":
        """Add rate limiting middleware."""
        config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            **kwargs,
        )
        return self.add(RateLimitMiddleware(config))

    def with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ) -> "MiddlewareStack":
        """Add retry middleware."""
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            **kwargs,
        )
        return self.add(RetryMiddleware(config))

    def with_logging(
        self,
        log_content: bool = False,
        **kwargs,
    ) -> "MiddlewareStack":
        """Add logging middleware."""
        config = LoggingConfig(log_content=log_content, **kwargs)
        return self.add(LoggingMiddleware(config))

    def build(self) -> List[Middleware]:
        """Build the middleware list."""
        return list(self._middleware)


def create_default_middleware() -> List[Middleware]:
    """Create default middleware stack."""
    return (
        MiddlewareStack()
        .with_logging()
        .with_caching(ttl=300)
        .with_rate_limiting(requests_per_minute=60)
        .with_retry(max_retries=3)
        .build()
    )
