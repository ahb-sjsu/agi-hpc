"""
Reusable gRPC server utilities for AGI-HPC services.

Features:
    - Configurable port, max workers, and message sizes
    - Optional TLS (server-side)
    - Optional interceptors (logging, auth, tracing, etc.)
    - Optional server reflection for debugging and tooling
    - Graceful shutdown helper

Usage (basic):

    from agi.core.api.grpc_server import GRPCServer, GRPCServerConfig
    from agi.proto_gen import plan_pb2_grpc

    cfg = GRPCServerConfig(port=50051)
    server = GRPCServer(cfg)

    server.add_servicer(
        servicer=MyPlanService(),
        add_fn=plan_pb2_grpc.add_PlanServiceServicer_to_server,
    )

    server.start()
    server.wait()

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
    # Reflection is optional; only used if available and enabled.
    from grpc_reflection.v1alpha import reflection
except Exception:  # pragma: no cover - optional dependency
    reflection = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class GRPCServerConfig:
    """
    Configuration for a gRPC server instance.

    You can also configure this via environment variables:

        AGI_GRPC_PORT
        AGI_GRPC_MAX_WORKERS
        AGI_GRPC_MAX_MESSAGE_MB

    TLS (optional):

        cert_chain_file: path to PEM-encoded server certificate (and chain)
        private_key_file: path to PEM-encoded private key

    Reflection (optional):

        enable_reflection: if True and grpc_reflection is installed, reflection will
                           be enabled for the given service_names.
    """

    port: int = field(default_factory=lambda: int(os.getenv("AGI_GRPC_PORT", "50051")))
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("AGI_GRPC_MAX_WORKERS", "32"))
    )
    max_message_mb: int = field(
        default_factory=lambda: int(os.getenv("AGI_GRPC_MAX_MESSAGE_MB", "64"))
    )

    # TLS config (optional)
    cert_chain_file: Optional[str] = None
    private_key_file: Optional[str] = None

    # Reflection config (optional)
    enable_reflection: bool = False
    reflection_service_names: Optional[Sequence[str]] = None

    # Interceptors
    interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None

    @property
    def max_message_bytes(self) -> int:
        return self.max_message_mb * 1024 * 1024

    @property
    def use_tls(self) -> bool:
        return bool(self.cert_chain_file and self.private_key_file)


class GRPCServer:
    """
    Thin wrapper around grpc.Server with sane defaults for AGI-HPC.

    Provides:
        - add_servicer(servicer, add_fn)
        - start()
        - wait()
        - add_signal_handlers()  (optional convenience for CLI services)
        - stop(grace)
    """

    def __init__(self, config: Optional[GRPCServerConfig] = None) -> None:
        self._cfg = config or GRPCServerConfig()

        options = [
            ("grpc.max_send_message_length", self._cfg.max_message_bytes),
            ("grpc.max_receive_message_length", self._cfg.max_message_bytes),
            # Optionally tune keepalive, etc., here if needed.
            # ("grpc.keepalive_time_ms", 60_000),
        ]

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._cfg.max_workers),
            options=options,
            interceptors=list(self._cfg.interceptors or []),
        )

        self._started = False
        self._lock = threading.Lock()

        logger.info(
            "[grpc] Initializing server on port %d (max_workers=%d, max_msg=%d MB)",
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
        """
        Register a gRPC servicer with the underlying server.

        Args:
            servicer: Implementation instance (e.g., PlanServiceServicer)
            add_fn: Generated add_*_to_server function from *_pb2_grpc
        """
        add_fn(servicer, self._server)
        logger.info(
            "[grpc] Registered servicer %s via %s",
            servicer.__class__.__name__,
            getattr(add_fn, "__name__", repr(add_fn)),
        )

    # ------------------------------------------------------------------ #
    # Reflection (optional)
    # ------------------------------------------------------------------ #

    def enable_reflection(
        self, additional_service_names: Optional[Iterable[str]] = None
    ) -> None:
        """
        Enable gRPC reflection, if the grpc_reflection package is installed.

        If GRPCServerConfig.enable_reflection was True and reflection was available,
        this will be called automatically from start() using the configured
        reflection_service_names.
        """
        if reflection is None:
            logger.warning(
                "[grpc] Reflection requested but grpc-reflection is not installed"
            )
            return

        if not self._cfg.reflection_service_names and not additional_service_names:
            logger.warning(
                "[grpc] Reflection requested but no service names provided. "
                "Did you forget to set reflection_service_names?"
            )
            return

        service_names: List[str] = []

        if self._cfg.reflection_service_names:
            service_names.extend(self._cfg.reflection_service_names)
        if additional_service_names:
            service_names.extend(additional_service_names)

        # Avoid duplicates
        service_names = sorted(set(service_names))

        # Required for reflection
        service_names.append(reflection.SERVICE_NAME)

        reflection.enable_server_reflection(service_names, self._server)
        logger.info(
            "[grpc] Reflection enabled for services: %s", ", ".join(service_names)
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> grpc.Server:
        """
        Start the server and bind to the configured port.

        Returns:
            The underlying grpc.Server instance.
        """
        with self._lock:
            if self._started:
                logger.warning("[grpc] start() called but server is already running")
                return self._server

            if self._cfg.use_tls:
                logger.info(
                    "[grpc] Starting TLS server on port %d (cert=%s)",
                    self._cfg.port,
                    self._cfg.cert_chain_file,
                )
                private_key = self._read_file(self._cfg.private_key_file)
                cert_chain = self._read_file(self._cfg.cert_chain_file)

                creds = grpc.ssl_server_credentials(
                    [(private_key, cert_chain)], require_client_auth=False
                )
                self._server.add_secure_port(f"[::]:{self._cfg.port}", creds)
            else:
                logger.info(
                    "[grpc] Starting INSECURE server on port %d", self._cfg.port
                )
                self._server.add_insecure_port(f"[::]:{self._cfg.port}")

            # Enable reflection if configured
            if self._cfg.enable_reflection:
                self.enable_reflection()

            self._server.start()
            self._started = True
            logger.info("[grpc] Server started on port %d", self._cfg.port)

        return self._server

    def wait(self) -> None:
        """
        Block until server termination.
        """
        logger.info("[grpc] Waiting for termination")
        self._server.wait_for_termination()

    def stop(self, grace: float = 5.0) -> None:
        """
        Initiate a graceful shutdown.

        Args:
            grace: Time in seconds to allow existing RPCs to complete.
        """
        with self._lock:
            if not self._started:
                return
            logger.info("[grpc] Stopping server (grace=%.1fs)...", grace)
            self._server.stop(grace)
            self._started = False

    def add_signal_handlers(self, *, use_default_signals: bool = True) -> None:
        """
        Install SIGINT/SIGTERM handlers that trigger a graceful shutdown.

        Call this from your service main() if you want Ctrl+C / kill to
        gracefully stop the server.
        """

        if not use_default_signals:
            return

        def _handler(signum, _frame):
            logger.info("[grpc] Received signal %s; initiating shutdown", signum)
            self.stop(grace=5.0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        logger.info("[grpc] Installed SIGINT/SIGTERM handlers")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_file(path: Optional[str]) -> bytes:
        if not path:
            raise ValueError("TLS file path is not set")
        with open(path, "rb") as f:
            return f.read()
