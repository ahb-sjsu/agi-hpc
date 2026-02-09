# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Shared LLM Infrastructure for AGI-HPC.

Provides unified LLM access across all subsystems with:
- Provider abstraction (Anthropic, OpenAI, Ollama)
- Middleware stack (retry, logging, rate limiting)
- Streaming support
- Async support

Usage:
    from agi.core.llm import LLMClient

    # Simple usage
    client = LLMClient()
    response = client.complete("What is 2+2?")
    print(response.content)

    # With specific provider
    client = LLMClient(provider="anthropic")

    # Streaming
    for chunk in client.stream("Tell me a story"):
        print(chunk.delta, end="")

Environment Variables:
    AGI_LLM_PROVIDER        Provider (anthropic, openai, ollama)
    AGI_LLM_MODEL           Model name
    AGI_LLM_API_KEY         API key
    AGI_LLM_BASE_URL        Base URL (optional)
    AGI_LLM_TEMPERATURE     Default temperature
    AGI_LLM_MAX_TOKENS      Default max tokens
"""

from agi.core.llm.types import (
    CompletionRequest,
    LLMResponse,
    Message,
    MessageRole,
    StreamChunk,
    StreamingResponse,
    Usage,
)
from agi.core.llm.config import LLMConfig, ProviderConfig, get_default_model
from agi.core.llm.client import (
    LLMClient,
    create_client,
    get_provider,
)
from agi.core.llm.middleware import (
    CachingMiddleware,
    CacheConfig,
    RateLimitMiddleware,
    RateLimitConfig,
    RateLimitExceeded,
    RetryMiddleware,
    RetryConfig,
    FallbackMiddleware,
    FallbackConfig,
    LoggingMiddleware,
    LoggingConfig,
    MiddlewareStack,
    create_default_middleware,
)

# Re-export providers
from agi.core.llm.providers import (
    LLMProvider,
    AsyncLLMProvider,
    BaseProvider,
    BaseAsyncProvider,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
)

__all__ = [
    # Main client
    "LLMClient",
    "create_client",
    "get_provider",
    # Types
    "CompletionRequest",
    "LLMResponse",
    "Message",
    "MessageRole",
    "StreamChunk",
    "StreamingResponse",
    "Usage",
    # Config
    "LLMConfig",
    "ProviderConfig",
    "get_default_model",
    # Middleware
    "CachingMiddleware",
    "CacheConfig",
    "RateLimitMiddleware",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RetryMiddleware",
    "RetryConfig",
    "FallbackMiddleware",
    "FallbackConfig",
    "LoggingMiddleware",
    "LoggingConfig",
    "MiddlewareStack",
    "create_default_middleware",
    # Providers
    "LLMProvider",
    "AsyncLLMProvider",
    "BaseProvider",
    "BaseAsyncProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
]
