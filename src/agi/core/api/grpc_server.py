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

"""
Reusable gRPC server utilities for AGI-HPC services.

Features:
    - Configurable port, max workers, and message sizes
    - Optional TLS (server-side)
    - Optional interceptors (logging, auth, tracing, etc.)
    - Optional reflection
    - Graceful shutdown helpers
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from concurrent import futures
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import grpc

try:
    from grpc_reflection.v1alpha import reflection
except Exception:  # pragma: no cover
    reflection = None


logger = logging.getLogger(__name__)


@dataclass
class GRPCServerConfig:
    """
    Configuration container for GRPCServer.
    """

    port: int = field(default_factory=lambda: int(os.getenv("AGI_GRPC_PORT", "50051")))
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("AGI_GRPC_MAX_WORKERS", "32"))
    )
    max_message_mb: int = field(
        default_factory=lambda: int(os.getenv("AGI_GRPC_MAX_MESSAGE_MB", "64"))
    )

    cert_chain_file: Optional[str] = None
    private_key_file: Optional[str] = None

    enable_reflection: bool = False
    reflection_service_names: Optional[Sequence[str]] = None

    interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None

    @property
    def max_message_bytes(self) -> int:
        return self.max_message_mb * 1024 * 1024

    @property
    def use_tls(self) -> bool:
        return bool(self.cert_chain_file and self.private_key_file)


class GRPCServer:
    """
    Wrapper around grpc.Server with sane defaults.
    """

    def __init__(self, config: Optional[GRPCServerConfig] = None) -> None:
        self._cfg = config or GRPCServerConfig()

        options = [
            ("grpc.max_send_message_length", self._cfg.max_message_bytes),
            ("grpc.max_receive_message_length", self._cfg.max_message_bytes),
        ]

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._cfg.max_workers),
            options=options,
            interceptors=list(self._cfg.interceptors or []),
        )

        self._started = False
        self._lock = threading.Lock()

        logger.info(
            "[grpc] init port=%d workers=%d max_msg=%dMB",
            self._cfg.port,
            self._cfg.max_workers,
            self._cfg.max_message_mb,
        )

    # ------------------------------------------------------------------ #
    # Servicer registration
    # ------------------------------------------------------------------ #

    def add_servicer(
        self,
        servicer: object,
        add_fn: Callable[[object, grpc.Server], None],
    ) -> None:
        add_fn(servicer, self._server)
        logger.info(
            "[grpc] servicer=%s added via %s",
            servicer.__class__.__name__,
            getattr(add_fn, "__name__", repr(add_fn)),
        )

    # ------------------------------------------------------------------ #
    # Reflection
    # ------------------------------------------------------------------ #

    def enable_reflection(
        self,
        additional_service_names: Optional[Iterable[str]] = None,
    ) -> None:
        if reflection is None:
            logger.warning("[grpc] reflection requested but not installed")
            return

        base_names = list(self._cfg.reflection_service_names or [])
        extra_names = list(additional_service_names or [])

        services = sorted(set(base_names + extra_names))
        if not services:
            logger.warning("[grpc] no reflection service names provided")
            return

        services.append(reflection.SERVICE_NAME)

        reflection.enable_server_reflection(services, self._server)
        logger.info(
            "[grpc] reflection enabled for services=%s",
            ", ".join(services),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> grpc.Server:
        with self._lock:
            if self._started:
                logger.warning("[grpc] server already running")
                return self._server

            if self._cfg.use_tls:
                logger.info(
                    "[grpc] starting TLS server on port %d",
                    self._cfg.port,
                )
                private_key = self._read_file(self._cfg.private_key_file)
                cert_chain = self._read_file(self._cfg.cert_chain_file)
                creds = grpc.ssl_server_credentials([(private_key, cert_chain)])
                self._server.add_secure_port(f"[::]:{self._cfg.port}", creds)
            else:
                logger.info(
                    "[grpc] starting INSECURE server on port %d",
                    self._cfg.port,
                )
                self._server.add_insecure_port(f"[::]:{self._cfg.port}")

            if self._cfg.enable_reflection:
                self.enable_reflection()

            self._server.start()
            self._started = True

            logger.info("[grpc] server started on port %d", self._cfg.port)

        return self._server

    def wait(self) -> None:
        logger.info("[grpc] waiting for termination")
        self._server.wait_for_termination()

    def stop(self, grace: float = 5.0) -> None:
        with self._lock:
            if not self._started:
                return
            logger.info("[grpc] stopping server grace=%.1fs", grace)
            self._server.stop(grace)
            self._started = False

    def add_signal_handlers(self, *, use_default_signals: bool = True) -> None:
        if not use_default_signals:
            return

        def _handler(signum, _frame) -> None:
            logger.info("[grpc] received signal=%s shutting down", signum)
            self.stop(grace=5.0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        logger.info("[grpc] installed SIGINT/SIGTERM handlers")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_file(path: Optional[str]) -> bytes:
        if not path:
            raise ValueError("TLS file path is not set")
        with open(path, "rb") as f:
            return f.read()
