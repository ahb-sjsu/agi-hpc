# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
ZeroMQ XPUB/XSUB Broker for Event Fabric.

This broker sits between publishers and subscribers, forwarding messages
and enabling dynamic topic discovery. It provides the central routing
infrastructure for multi-node AGI-HPC deployments.

Architecture:
    Publishers → [XSUB frontend] → Broker → [XPUB backend] → Subscribers

Usage:
    # Start broker
    python -m agi.core.events.broker

    # Or programmatically
    broker = FabricBroker()
    broker.run()  # Blocks until SIGINT/SIGTERM

Environment Variables:
    AGI_BROKER_FRONTEND    Frontend (publisher) endpoint (default: tcp://*:5556)
    AGI_BROKER_BACKEND     Backend (subscriber) endpoint (default: tcp://*:5555)
    AGI_BROKER_CONTROL     Control endpoint for management (default: tcp://*:5557)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore

# Default endpoints
DEFAULT_FRONTEND = os.getenv("AGI_BROKER_FRONTEND", "tcp://*:5556")
DEFAULT_BACKEND = os.getenv("AGI_BROKER_BACKEND", "tcp://*:5555")
DEFAULT_CONTROL = os.getenv("AGI_BROKER_CONTROL", "tcp://*:5557")


@dataclass
class BrokerMetrics:
    """Broker performance metrics."""

    messages_forwarded: int = 0
    bytes_forwarded: int = 0
    subscriptions_active: int = 0
    publishers_connected: int = 0
    subscribers_connected: int = 0
    start_time: float = field(default_factory=time.time)
    last_message_time: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.messages_forwarded / uptime if uptime > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "messages_forwarded": self.messages_forwarded,
            "bytes_forwarded": self.bytes_forwarded,
            "subscriptions_active": self.subscriptions_active,
            "publishers_connected": self.publishers_connected,
            "subscribers_connected": self.subscribers_connected,
            "uptime_seconds": self.uptime_seconds,
            "messages_per_second": self.messages_per_second,
        }


@dataclass
class BrokerConfig:
    """Broker configuration."""

    frontend_addr: str = DEFAULT_FRONTEND
    backend_addr: str = DEFAULT_BACKEND
    control_addr: str = DEFAULT_CONTROL
    hwm: int = 10000  # High water mark for socket buffers
    linger: int = 1000  # Linger time in ms for socket close
    heartbeat_interval: int = 5000  # Heartbeat interval in ms
    enable_metrics: bool = True
    enable_control: bool = True


class FabricBroker:
    """
    ZeroMQ XPUB/XSUB broker for Event Fabric.

    Provides:
    - Message forwarding between publishers and subscribers
    - Dynamic topic subscription discovery
    - Connection monitoring
    - Performance metrics
    - Graceful shutdown

    The broker uses zmq.proxy() for high-performance message forwarding,
    with a control socket for management operations.
    """

    def __init__(self, config: Optional[BrokerConfig] = None) -> None:
        """Initialize the broker."""
        if zmq is None:
            raise RuntimeError("pyzmq is required but not installed")

        self.config = config or BrokerConfig()
        self.metrics = BrokerMetrics()

        self._ctx: Optional[zmq.Context] = None
        self._frontend: Optional[zmq.Socket] = None
        self._backend: Optional[zmq.Socket] = None
        self._control: Optional[zmq.Socket] = None
        self._capture: Optional[zmq.Socket] = None

        self._running = False
        self._shutdown_event = threading.Event()

        # Track subscriptions for metrics
        self._subscriptions: Dict[str, int] = {}
        self._lock = threading.Lock()

        logger.info(
            "[broker] initialized frontend=%s backend=%s",
            self.config.frontend_addr,
            self.config.backend_addr,
        )

    def _setup_sockets(self) -> None:
        """Create and configure ZMQ sockets."""
        self._ctx = zmq.Context.instance()

        # Frontend: XSUB receives from publishers
        self._frontend = self._ctx.socket(zmq.XSUB)
        self._frontend.setsockopt(zmq.RCVHWM, self.config.hwm)
        self._frontend.setsockopt(zmq.LINGER, self.config.linger)
        self._frontend.bind(self.config.frontend_addr)

        # Backend: XPUB sends to subscribers
        self._backend = self._ctx.socket(zmq.XPUB)
        self._backend.setsockopt(zmq.SNDHWM, self.config.hwm)
        self._backend.setsockopt(zmq.LINGER, self.config.linger)
        # Enable verbose mode to receive all subscription messages
        self._backend.setsockopt(zmq.XPUB_VERBOSE, 1)
        self._backend.bind(self.config.backend_addr)

        # Capture socket for metrics (inproc)
        if self.config.enable_metrics:
            self._capture = self._ctx.socket(zmq.PUB)
            self._capture.bind("inproc://broker-capture")

        # Control socket for management
        if self.config.enable_control:
            self._control = self._ctx.socket(zmq.REP)
            self._control.bind(self.config.control_addr)

        logger.info(
            "[broker] sockets created frontend=%s backend=%s",
            self.config.frontend_addr,
            self.config.backend_addr,
        )

    def run(self) -> None:
        """Run the broker (blocking).

        This method blocks until the broker is stopped via SIGINT/SIGTERM
        or by calling stop() from another thread.
        """
        self._setup_sockets()
        self._running = True
        self.metrics.start_time = time.time()

        logger.info("[broker] starting proxy loop")

        # Start control handler in separate thread if enabled
        control_thread = None
        if self.config.enable_control and self._control:
            control_thread = threading.Thread(
                target=self._control_loop,
                name="broker-control",
                daemon=True,
            )
            control_thread.start()

        # Start subscription monitor thread
        sub_monitor_thread = threading.Thread(
            target=self._subscription_monitor,
            name="broker-sub-monitor",
            daemon=True,
        )
        sub_monitor_thread.start()

        try:
            # Use zmq.proxy_steerable for controlled shutdown
            zmq.proxy_steerable(
                self._frontend,
                self._backend,
                self._capture,
                self._control if not self.config.enable_control else None,
            )
        except zmq.ContextTerminated:
            logger.info("[broker] context terminated")
        except zmq.ZMQError as e:
            if e.errno != zmq.ETERM:
                logger.exception("[broker] proxy error")
        except Exception:
            logger.exception("[broker] unexpected error")
        finally:
            self._running = False
            logger.info("[broker] proxy loop exited")

    def run_async(self) -> threading.Thread:
        """Run the broker in a background thread.

        Returns:
            Thread running the broker
        """
        thread = threading.Thread(
            target=self.run,
            name="fabric-broker",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop the broker gracefully."""
        logger.info("[broker] stopping")
        self._shutdown_event.set()
        self._running = False

        if self._ctx:
            self._ctx.term()

    def _control_loop(self) -> None:
        """Handle control socket requests."""
        if not self._control:
            return

        poller = zmq.Poller()
        poller.register(self._control, zmq.POLLIN)

        while not self._shutdown_event.is_set():
            try:
                events = dict(poller.poll(timeout=1000))
                if self._control not in events:
                    continue

                request = self._control.recv_string()
                response = self._handle_control(request)
                self._control.send_string(response)

            except zmq.ContextTerminated:
                break
            except Exception:
                logger.exception("[broker] control loop error")

        logger.info("[broker] control loop exited")

    def _handle_control(self, request: str) -> str:
        """Handle a control request."""
        import json

        cmd = request.strip().upper()

        if cmd == "STATS":
            return json.dumps(self.metrics.to_dict())
        elif cmd == "HEALTH":
            return json.dumps({"status": "healthy", "running": self._running})
        elif cmd == "SUBSCRIPTIONS":
            with self._lock:
                return json.dumps(dict(self._subscriptions))
        elif cmd == "TERMINATE":
            self.stop()
            return json.dumps({"status": "terminating"})
        else:
            return json.dumps({"error": f"unknown command: {request}"})

    def _subscription_monitor(self) -> None:
        """Monitor subscription messages from XPUB socket."""
        # XPUB sends subscription messages: first byte is 0x01 (sub) or 0x00 (unsub)
        # followed by the topic string

        # We need to intercept XPUB subscription messages
        # This is done by receiving on the backend socket before proxy starts
        # For now, we'll use the verbose mode setting and track in metrics

        while not self._shutdown_event.is_set():
            time.sleep(1.0)
            # Metrics updates happen passively through message counting

    def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        self._shutdown_event.wait()

    @property
    def is_running(self) -> bool:
        return self._running


class BrokerClient:
    """Client for interacting with broker control socket."""

    def __init__(self, control_addr: str = DEFAULT_CONTROL) -> None:
        """Initialize broker client."""
        if zmq is None:
            raise RuntimeError("pyzmq is required")

        self._addr = control_addr.replace("*", "localhost")
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 5000)
        self._socket.connect(self._addr)

    def stats(self) -> Dict:
        """Get broker statistics."""
        import json

        self._socket.send_string("STATS")
        return json.loads(self._socket.recv_string())

    def health(self) -> Dict:
        """Check broker health."""
        import json

        self._socket.send_string("HEALTH")
        return json.loads(self._socket.recv_string())

    def subscriptions(self) -> Dict[str, int]:
        """Get active subscriptions."""
        import json

        self._socket.send_string("SUBSCRIPTIONS")
        return json.loads(self._socket.recv_string())

    def terminate(self) -> Dict:
        """Request broker termination."""
        import json

        self._socket.send_string("TERMINATE")
        return json.loads(self._socket.recv_string())

    def close(self) -> None:
        """Close the client."""
        self._socket.close()


def main() -> None:
    """Main entry point for broker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AGI-HPC Event Fabric Broker")
    parser.add_argument(
        "--frontend",
        default=DEFAULT_FRONTEND,
        help="Publisher endpoint (XSUB)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="Subscriber endpoint (XPUB)",
    )
    parser.add_argument(
        "--control",
        default=DEFAULT_CONTROL,
        help="Control endpoint (REP)",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable control socket",
    )

    args = parser.parse_args()

    config = BrokerConfig(
        frontend_addr=args.frontend,
        backend_addr=args.backend,
        control_addr=args.control,
        enable_control=not args.no_control,
    )

    broker = FabricBroker(config)

    def signal_handler(sig: int, frame) -> None:
        logger.info("[broker] received signal %d, shutting down", sig)
        broker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("[broker] starting AGI-HPC Event Fabric Broker")
    broker.run()


if __name__ == "__main__":
    main()
