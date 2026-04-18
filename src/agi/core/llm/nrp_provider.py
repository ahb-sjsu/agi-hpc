"""NRP Managed LLM Provider — routes Divine Council to NRP Nautilus API.

Replaces local llama.cpp servers with the NRP managed LLM service,
freeing both local GPUs for compute while upgrading model capabilities:

  Local (llama.cpp)              NRP Managed API
  ────────────────               ───────────────
  Id: Qwen 3 32B (GPU 1)    ->  qwen3: Qwen 3.5 397B
  Superego: Gemma 4 31B (GPU 0) ->  gemma-4-e4b: Gemma 4
  Ego: Gemma 4 26B MoE (CPU)   ->  kimi: Kimi-K2.5 1T

Usage:
    provider = NRPProvider.from_config("configs/nrp_llm_config.yaml")
    response = await provider.chat("id", messages=[...])
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentConfig:
    model: str
    role: str
    temperature: float = 0.5
    max_tokens: int = 4096
    extra_body: dict | None = None


class NRPProvider:
    """OpenAI-compatible client for NRP managed LLM service."""

    def __init__(self, base_url: str, api_key: str,
                 agents: dict[str, AgentConfig] | None = None,
                 max_concurrent: int = 5,
                 delay_between_calls: float = 0.5):
        self.base_url = base_url
        self.api_key = api_key
        self.agents = agents or {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._delay = delay_between_calls
        self._last_call = 0.0
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @classmethod
    def from_config(cls, config_path: str) -> NRPProvider:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        nrp = cfg.get("nrp_llm", {})
        api_key = nrp.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")

        agents = {}
        for name, acfg in nrp.get("agents", {}).items():
            agents[name] = AgentConfig(
                model=acfg["model"],
                role=acfg.get("role", ""),
                temperature=acfg.get("temperature", 0.5),
                max_tokens=acfg.get("max_tokens", 4096),
                extra_body=acfg.get("extra_body"),
            )

        rate = nrp.get("rate_limit", {})
        return cls(
            base_url=nrp["base_url"],
            api_key=api_key,
            agents=agents,
            max_concurrent=rate.get("max_concurrent", 5),
            delay_between_calls=rate.get("delay_between_calls_s", 0.5),
        )

    def chat(self, agent_name: str, messages: list[dict],
             system: str | None = None, **kwargs) -> str:
        """Synchronous chat with a named agent."""
        acfg = self.agents.get(agent_name)
        if not acfg:
            raise ValueError(f"Unknown agent: {agent_name}. "
                           f"Available: {list(self.agents.keys())}")

        # Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

        client = self._get_client()

        # Build messages
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        elif acfg.role:
            full_messages.append({"role": "system", "content": acfg.role})
        full_messages.extend(messages)

        # Build kwargs
        call_kwargs: dict[str, Any] = {
            "model": acfg.model,
            "messages": full_messages,
            "temperature": kwargs.get("temperature", acfg.temperature),
            "max_tokens": kwargs.get("max_tokens", acfg.max_tokens),
        }
        if acfg.extra_body:
            call_kwargs["extra_body"] = acfg.extra_body

        try:
            r = client.chat.completions.create(**call_kwargs)
            self._last_call = time.time()
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"[NRP LLM Error: {e}]"

    async def achat(self, agent_name: str, messages: list[dict],
                    system: str | None = None, **kwargs) -> str:
        """Async chat with rate limiting."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.chat(agent_name, messages, system, **kwargs)
            )

    def list_agents(self) -> dict[str, str]:
        """Return agent name -> model mapping."""
        return {name: f"{acfg.model} ({acfg.role})"
                for name, acfg in self.agents.items()}

    def test_connection(self) -> dict[str, str]:
        """Test each agent with a simple prompt."""
        results = {}
        for name in self.agents:
            try:
                response = self.chat(name, [{"role": "user", "content": "Say hello in one word."}])
                results[name] = f"OK: {response.strip()[:50]}"
            except Exception as e:
                results[name] = f"FAIL: {e}"
        return results


if __name__ == "__main__":
    import sys
    os.environ.setdefault("NRP_LLM_TOKEN", "qXG2BOVwEJFeQ0zua6Hbqk91atcmUUKN")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/nrp_llm_config.yaml"
    provider = NRPProvider.from_config(config_path)

    print("Agents configured:")
    for name, desc in provider.list_agents().items():
        print(f"  {name}: {desc}")

    print("\nTesting connections...")
    results = provider.test_connection()
    for name, result in results.items():
        print(f"  {name}: {result}")
