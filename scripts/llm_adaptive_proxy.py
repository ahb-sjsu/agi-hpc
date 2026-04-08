#!/usr/bin/env python3
"""Adaptive LLM proxy that switches between fast/long-context KV cache modes.

Sits between clients and llama-server, transparently switching between:
  - FAST mode:  fp16 KV cache, 8K context, ~4.2 tok/s
  - LONG mode:  q8_0 KV cache, 14K context, ~2.1 tok/s

Switching logic:
  - Estimates prompt token count (~4 chars/token)
  - If prompt fits in 8K budget: use FAST mode (no switch needed)
  - If prompt exceeds threshold: switch to LONG mode, serve, switch back
  - If LONG mode is already active: serve immediately
  - Cooldown: stays in LONG mode for 60s after last long request to
    avoid thrashing if multiple long requests arrive in sequence

The switch takes ~20s (model reload). This is acceptable because:
  - Long-context requests are rare (most RAG queries fit in 8K)
  - When they do occur, the user benefits from 14K context
  - The proxy is transparent — clients see the same API

Usage:
    python llm_adaptive_proxy.py                  # default settings
    python llm_adaptive_proxy.py --threshold 6000  # switch at 6K tokens
    python llm_adaptive_proxy.py --port 8090       # proxy port

Clients connect to the proxy port (default 8090) instead of 8080.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import threading
import time
from enum import Enum
from typing import Optional

import requests
from flask import Flask, Response, request as flask_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("llm-proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get(
    "LLM_MODEL",
    "/home/claude/models/Qwen2.5-72B-Instruct-Q5_K_M/"
    "qwen2.5-72b-instruct-q5_k_m-00001-of-00014.gguf",
)
LLAMA_BIN = os.environ.get(
    "LLAMA_BIN",
    "/home/claude/llama.cpp/build/bin/llama-server",
)
LLM_PORT = 8080
CHAT_PATH = "/home/claude/atlas-chat"
THREADS = 24
GPU_ID = 0

CHARS_PER_TOKEN = 4  # conservative estimate


class Mode(Enum):
    FAST = "fast"  # fp16, 8K ctx
    LONG = "long"  # q8_0, 14K ctx


MODE_CONFIGS = {
    Mode.FAST: {
        "ctx_size": 8192,
        "cache_k": "f16",
        "cache_v": "f16",
        "label": "fp16/8K (fast)",
    },
    Mode.LONG: {
        "ctx_size": 14336,
        "cache_k": "q8_0",
        "cache_v": "q8_0",
        "label": "q8_0/14K (long-context)",
    },
}


# ---------------------------------------------------------------------------
# Server manager
# ---------------------------------------------------------------------------


class LLMServerManager:
    """Manages the llama-server process lifecycle."""

    def __init__(self) -> None:
        self._mode: Optional[Mode] = None
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._last_long_request: float = 0
        self._switch_count: int = 0

    @property
    def mode(self) -> Optional[Mode]:
        return self._mode

    @property
    def is_ready(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            r = requests.get(
                f"http://localhost:{LLM_PORT}/health", timeout=2
            )
            return r.status_code == 200
        except Exception:
            return False

    def start(self, mode: Mode) -> bool:
        """Start or switch to the given mode. Returns True when ready."""
        with self._lock:
            if self._mode == mode and self.is_ready:
                return True

            log.info("Switching to %s...", MODE_CONFIGS[mode]["label"])
            self._kill()

            cfg = MODE_CONFIGS[mode]
            cmd = [
                LLAMA_BIN,
                "--model", MODEL,
                "--host", "0.0.0.0",
                "--port", str(LLM_PORT),
                "--ctx-size", str(cfg["ctx_size"]),
                "--threads", str(THREADS),
                "--path", CHAT_PATH,
            ]
            if cfg["cache_k"] != "f16":
                cmd.extend(["--cache-type-k", cfg["cache_k"]])
            if cfg["cache_v"] != "f16":
                cmd.extend(["--cache-type-v", cfg["cache_v"]])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._mode = mode
            self._switch_count += 1

            # Wait for ready
            for _ in range(60):  # 5 min max
                time.sleep(5)
                if self._process.poll() is not None:
                    log.error("Server exited during startup")
                    return False
                if self.is_ready:
                    log.info(
                        "Server ready in %s mode (switch #%d)",
                        mode.value,
                        self._switch_count,
                    )
                    return True

            log.error("Server failed to become ready in 5 minutes")
            return False

    def _kill(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
            self._mode = None
            time.sleep(2)  # let GPU memory release

    def record_long_request(self) -> None:
        self._last_long_request = time.time()

    def should_switch_back(self, cooldown: float = 60.0) -> bool:
        """Whether enough time has passed since last long request."""
        if self._mode != Mode.LONG:
            return False
        return time.time() - self._last_long_request > cooldown

    def stats(self) -> dict:
        return {
            "mode": self._mode.value if self._mode else "none",
            "ready": self.is_ready,
            "switch_count": self._switch_count,
            "last_long_request": self._last_long_request,
        }

    def shutdown(self) -> None:
        self._kill()


# ---------------------------------------------------------------------------
# Proxy app
# ---------------------------------------------------------------------------


def create_proxy(
    threshold_tokens: int = 6000,
    cooldown_seconds: float = 60.0,
    proxy_port: int = 8090,
) -> Flask:
    app = Flask(__name__)
    manager = LLMServerManager()

    # Start in FAST mode
    manager.start(Mode.FAST)

    # Background thread to switch back to FAST after cooldown
    def _cooldown_loop():
        while True:
            time.sleep(10)
            if manager.should_switch_back(cooldown_seconds):
                log.info("Cooldown expired, switching back to FAST mode")
                manager.start(Mode.FAST)

    t = threading.Thread(target=_cooldown_loop, daemon=True)
    t.start()

    def _estimate_tokens(data: dict) -> int:
        """Estimate total prompt tokens from a chat completion request."""
        messages = data.get("messages", [])
        total_chars = sum(
            len(m.get("content", "")) for m in messages
        )
        return total_chars // CHARS_PER_TOKEN

    @app.route("/health", methods=["GET"])
    def health():
        return json.dumps(manager.stats()), 200

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        data = flask_request.get_json(force=True)
        est_tokens = _estimate_tokens(data)

        # Decide mode
        need_long = est_tokens > threshold_tokens
        target = Mode.LONG if need_long else Mode.FAST

        if need_long:
            manager.record_long_request()

        # Switch if needed
        current = manager.mode
        if current != target:
            log.info(
                "Request needs %d tokens, switching %s -> %s",
                est_tokens,
                current.value if current else "none",
                target.value,
            )
            if not manager.start(target):
                return json.dumps(
                    {"error": "Failed to switch LLM mode"}
                ), 503

        # Forward request to llama-server
        try:
            resp = requests.post(
                f"http://localhost:{LLM_PORT}/v1/chat/completions",
                json=data,
                timeout=300,
                stream=data.get("stream", False),
            )

            if data.get("stream", False):
                return Response(
                    resp.iter_content(chunk_size=None),
                    content_type=resp.headers.get("Content-Type"),
                    status=resp.status_code,
                )

            return resp.text, resp.status_code, {
                "Content-Type": "application/json",
                "X-LLM-Mode": target.value,
                "X-Est-Tokens": str(est_tokens),
            }
        except requests.exceptions.RequestException as e:
            log.error("Forward failed: %s", e)
            return json.dumps({"error": str(e)}), 502

    # Pass through other endpoints
    @app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
    def proxy_passthrough(path):
        try:
            resp = requests.request(
                method=flask_request.method,
                url=f"http://localhost:{LLM_PORT}/{path}",
                headers={
                    k: v
                    for k, v in flask_request.headers
                    if k.lower() != "host"
                },
                data=flask_request.get_data(),
                timeout=300,
            )
            return resp.text, resp.status_code
        except Exception as e:
            return json.dumps({"error": str(e)}), 502

    @app.route("/", methods=["GET"])
    def index():
        try:
            resp = requests.get(
                f"http://localhost:{LLM_PORT}/",
                timeout=5,
            )
            return resp.text, resp.status_code, {
                "Content-Type": resp.headers.get("Content-Type", "text/html")
            }
        except Exception as e:
            return json.dumps({"error": str(e)}), 502

    def _shutdown(signum, frame):
        log.info("Shutting down...")
        manager.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive LLM proxy")
    parser.add_argument(
        "--threshold", type=int, default=6000,
        help="Token threshold for switching to LONG mode (default: 6000)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=60.0,
        help="Seconds to stay in LONG mode after last long request (default: 60)",
    )
    parser.add_argument(
        "--port", type=int, default=8090,
        help="Proxy listen port (default: 8090)",
    )
    args = parser.parse_args()

    app = create_proxy(
        threshold_tokens=args.threshold,
        cooldown_seconds=args.cooldown,
        proxy_port=args.port,
    )
    log.info(
        "Adaptive LLM proxy on :%d (threshold=%d tokens, cooldown=%ds)",
        args.port,
        args.threshold,
        args.cooldown,
    )
    app.run(host="0.0.0.0", port=args.port, threaded=True)
