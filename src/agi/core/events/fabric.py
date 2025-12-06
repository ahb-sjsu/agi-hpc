"""
Event fabric abstraction for AGI-HPC.

Backends:
    - local : in-process pub/sub (no networking, best for tests and single-process dev)
    - zmq   : ZeroMQ PUB/SUB over TCP (simple multi-process / multi-node)
    - ucx   : UCX (via ucx-py / ucp) for low-latency HPC inter-node transport

Mode selection is via environment variable:

    AGI_FABRIC_MODE = "local" | "zmq" | "ucx"

Additional configuration:

    # ZMQ backend
    AGI_FABRIC_PUB_ENDPOINT = "tcp://fabric:5556"
    AGI_FABRIC_SUB_ENDPOINT = "tcp://fabric:5555"

    # UCX backend (client -> broker/server)
    AGI_FABRIC_UCX_ENDPOINT = "tcp://fabric:13337"

Public API:

    fabric = EventFabric()
    fabric.subscribe("topic.name", handler: Callable[[dict], None])
    fabric.publish("topic.name", {"payload": 1})
    fabric.close()

All messages are JSON-encoded dicts, transported as UTF-8 bytes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict], None]

# ---------------------------------------------------------------------------
# Optional dependencies (pyzmq, ucx-py)
# ---------------------------------------------------------------------------

try:  # ZeroMQ backend
    import zmq  # type: ignore[import]
except Exception:  # pragma: no cover - optional
    zmq = None  # type: ignore[assignment]

try:  # UCX backend
    import ucp  # type: ignore[import]
except Exception:  # pragma: no cover - optional
    ucp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODE = os.getenv("AGI_FABRIC_MODE", "local").lower()

# ZMQ
DEFAULT_PUB_ENDPOINT = os.getenv("AGI_FABRIC_PUB_ENDPOINT", "tcp://fabric:5556")
DEFAULT_SUB_ENDPOINT = os.getenv("AGI_FABRIC_SUB_ENDPOINT", "tcp://fabric:5555")

# UCX
DEFAULT_UCX_ENDPOINT = os.getenv("AGI_FABRIC_UCX_ENDPOINT", "tcp://fabric:13337")
DEFAULT_IDENTITY = os.getenv("AGI_FABRIC_IDENTITY", "node")


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FabricBackend(Protocol):
    def publish(self, topic: str, message: dict) -> None:  # pragma: no cover - protocol
        ...

    def subscribe(self, topic: str, handler: EventHandler) -> None:  # pragma: no cover
        ...

    def close(self) -> None:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Local in-process backend
# ---------------------------------------------------------------------------

class LocalBackend:
    """
    Simple in-process pub/sub.

    - No networking
    - Handlers are called synchronously in the caller's thread
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        logger.info("[fabric][local] initialized")

    def publish(self, topic: str, message: dict) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        logger.debug(
            "[fabric][local] publish topic=%s handlers=%d", topic, len(handlers)
        )
        for fn in handlers:
            try:
                fn(message)
            except Exception:  # pragma: no cover - defensive
                logger.exception("[fabric][local] handler error for topic=%s", topic)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        logger.info(
            "[fabric][local] subscribed topic=%s handler=%s",
            topic,
            getattr(handler, "__name__", repr(handler)),
        )

    def close(self) -> None:
        with self._lock:
            self._subscribers.clear()
        logger.info("[fabric][local] closed")


# ---------------------------------------------------------------------------
# ZeroMQ backend
# ---------------------------------------------------------------------------

class ZmqBackend:
    """
    ZeroMQ PUB/SUB backend.

    Topology assumption:
        - One or more 'broker' processes providing XPUB/XSUB bridge
        - Each node:
            * connects PUB socket to AGI_FABRIC_PUB_ENDPOINT
            * connects SUB socket to AGI_FABRIC_SUB_ENDPOINT

    Wire format: multipart message [topic, json_payload]
    """

    def __init__(
        self,
        pub_endpoint: str = DEFAULT_PUB_ENDPOINT,
        sub_endpoint: str = DEFAULT_SUB_ENDPOINT,
        identity: str = DEFAULT_IDENTITY,
    ) -> None:
        if zmq is None:
            raise RuntimeError(
                "ZMQ backend requested but 'pyzmq' is not installed. "
                "Add 'pyzmq' to dependencies."
            )

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._sub = self._ctx.socket(zmq.SUB)
        self._identity = identity
        self._pub.setsockopt_string(zmq.IDENTITY, identity)
        self._sub.setsockopt_string(zmq.IDENTITY, identity)

        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None

        self._pub.connect(pub_endpoint)
        self._sub.connect(sub_endpoint)

        # Start background receiver
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name=f"fabric-zmq-recv-{identity}",
            daemon=True,
        )
        self._recv_thread.start()

        logger.info(
            "[fabric][zmq] initialized id=%s pub=%s sub=%s",
            identity,
            pub_endpoint,
            sub_endpoint,
        )

    def publish(self, topic: str, message: dict) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        try:
            self._pub.send_multipart([topic.encode("utf-8"), payload])
            logger.debug(
                "[fabric][zmq] publish topic=%s size=%d", topic, len(payload)
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("[fabric][zmq] publish failed topic=%s", topic)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

        # Update SUB socket filter
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        logger.info(
            "[fabric][zmq] subscribed topic=%s handler=%s",
            topic,
            getattr(handler, "__name__", repr(handler)),
        )

    def close(self) -> None:
        self._stop_event.set()
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)

        try:
            self._sub.close(linger=0)
        except Exception:  # pragma: no cover
            logger.exception("[fabric][zmq] error closing SUB socket")

        try:
            self._pub.close(linger=0)
        except Exception:  # pragma: no cover
            logger.exception("[fabric][zmq] error closing PUB socket")

        # Do not terminate shared context; other sockets may exist.
        logger.info("[fabric][zmq] closed")

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)

        logger.info("[fabric][zmq] recv loop started id=%s", self._identity)

        while not self._stop_event.is_set():
            try:
                events = dict(poller.poll(timeout=1000))  # 1s
            except zmq.ZMQError as exc:  # pragma: no cover - defensive
                if self._stop_event.is_set():
                    break
                logger.exception("[fabric][zmq] poll error: %s", exc)
                continue

            if self._sub in events and events[self._sub] & zmq.POLLIN:
                try:
                    frames = self._sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except Exception:  # pragma: no cover
                    logger.exception("[fabric][zmq] recv_multipart failed")
                    continue

                if len(frames) != 2:
                    logger.warning(
                        "[fabric][zmq] ignoring malformed message frames=%d",
                        len(frames),
                    )
                    continue

                topic = frames[0].decode("utf-8", errors="ignore")
                try:
                    payload = json.loads(frames[1].decode("utf-8"))
                except Exception:  # pragma: no cover
                    logger.exception(
                        "[fabric][zmq] failed to decode JSON for topic=%s", topic
                    )
                    continue

                with self._lock:
                    handlers = list(self._subscribers.get(topic, []))

                logger.debug(
                    "[fabric][zmq] received topic=%s handlers=%d",
                    topic,
                    len(handlers),
                )

                for fn in handlers:
                    try:
                        fn(payload)
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "[fabric][zmq] handler error for topic=%s", topic
                        )

        logger.info("[fabric][zmq] recv loop exiting id=%s", self._identity)


# ---------------------------------------------------------------------------
# UCX backend (via ucx-py / ucp)
# ---------------------------------------------------------------------------

@dataclass
class _UcxClientState:
    endpoint_str: str
    ep: Optional["ucp.Endpoint"] = None  # type: ignore[name-defined]
    connected: bool = False


class UcxBackend:
    """
    UCX-based backend using ucx-py (ucp) for HPC inter-node transport.

    Topology:
        - Assumes a separate UCX server/broker listening on AGI_FABRIC_UCX_ENDPOINT
        - This backend connects as a client and sends/receives framed messages.

    Wire format:
        [4-byte little-endian length][topic utf-8][0x00][json utf-8]
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_UCX_ENDPOINT,
        identity: str = DEFAULT_IDENTITY,
    ) -> None:
        if ucp is None:
            raise RuntimeError(
                "UCX backend requested but 'ucx-py' (ucp) is not installed. "
                "Add 'ucx-py' to dependencies."
            )

        self._identity = identity
        self._state = _UcxClientState(endpoint_str=endpoint)
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Async machinery
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"fabric-ucx-loop-{identity}",
            daemon=True,
        )
        self._thread.start()

        # Initialize UCX and schedule connect + recv loop
        def _init_ucx() -> None:
            ucp.init()
        self._loop.call_soon_threadsafe(_init_ucx)
        asyncio.run_coroutine_threadsafe(self._connect_and_recv(), self._loop)

        logger.info("[fabric][ucx] initialized id=%s endpoint=%s", identity, endpoint)

    # Public API ---------------------------------------------------------

    def publish(self, topic: str, message: dict) -> None:
        """
        Schedule a UCX send on the background event loop.
        """
        data = json.dumps(message, separators=(",", ":")).encode("utf-8")
        wire = topic.encode("utf-8") + b"\0" + data
        size_bytes = len(wire).to_bytes(4, "little")
        payload = size_bytes + wire

        async def _send() -> None:
            if not self._state.connected or self._state.ep is None:
                logger.debug(
                    "[fabric][ucx] publish but not yet connected; dropping message"
                )
                return
            try:
                await self._state.ep.send(payload)
                logger.debug(
                    "[fabric][ucx] published topic=%s size=%d", topic, len(payload)
                )
            except Exception:  # pragma: no cover
                logger.exception("[fabric][ucx] send failed topic=%s", topic)

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        else:
            logger.warning("[fabric][ucx] event loop not running; dropping message")

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        logger.info(
            "[fabric][ucx] subscribed topic=%s handler=%s",
            topic,
            getattr(handler, "__name__", repr(handler)),
        )

    def close(self) -> None:
        self._stop_event.set()

        async def _shutdown() -> None:
            if self._state.ep is not None:
                try:
                    await self._state.ep.close()
                except Exception:  # pragma: no cover
                    logger.exception("[fabric][ucx] error closing endpoint")
            try:
                ucp.finalize()
            except Exception:  # pragma: no cover
                logger.exception("[fabric][ucx] error in ucp.finalize()")

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            try:
                fut.result(timeout=2.0)
            except Exception:  # pragma: no cover
                logger.exception("[fabric][ucx] error during shutdown")

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info("[fabric][ucx] closed id=%s", self._identity)

    # Internal async machinery ------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_and_recv(self) -> None:
        """
        Connect to UCX endpoint and start a recv loop.

        The remote side (broker/server) is expected to:
            - Accept connections
            - Fan-out messages to all peers
            - Use the same framing protocol.
        """
        # Parse endpoint of form "tcp://host:port"
        ep_str = self._state.endpoint_str
        if not ep_str.startswith("tcp://"):
            logger.warning(
                "[fabric][ucx] only tcp:// endpoints are currently supported; got %s",
                ep_str,
            )
        host_port = ep_str.replace("tcp://", "")
        try:
            host, port_str = host_port.split(":")
            port = int(port_str)
        except ValueError:
            logger.error("[fabric][ucx] invalid endpoint: %s", ep_str)
            return

        logger.info("[fabric][ucx] connecting to %s:%d", host, port)

        try:
            ep = await ucp.create_endpoint(host, port)
        except Exception:
            logger.exception("[fabric][ucx] failed to create endpoint")
            return

        self._state.ep = ep
        self._state.connected = True

        logger.info("[fabric][ucx] connected to %s:%d", host, port)

        # Receive loop
        while not self._stop_event.is_set():
            try:
                # First read 4-byte size
                size_buf = bytearray(4)
                await ep.recv(size_buf)
                size = int.from_bytes(size_buf, "little")
                if size <= 0 or size > 16 * 1024 * 1024:
                    logger.warning("[fabric][ucx] unreasonable message size=%d", size)
                    continue

                payload = bytearray(size)
                await ep.recv(payload)

                # Parse: topic\0json
                try:
                    topic_bytes, msg_bytes = payload.split(b"\0", 1)
                except ValueError:
                    logger.warning("[fabric][ucx] malformed payload; missing separator")
                    continue

                topic = topic_bytes.decode("utf-8", errors="ignore")
                try:
                    message = json.loads(msg_bytes.decode("utf-8"))
                except Exception:
                    logger.exception(
                        "[fabric][ucx] failed to decode JSON for topic=%s", topic
                    )
                    continue

                with self._lock:
                    handlers = list(self._subscribers.get(topic, []))

                logger.debug(
                    "[fabric][ucx] received topic=%s handlers=%d",
                    topic,
                    len(handlers),
                )

                for fn in handlers:
                    try:
                        fn(message)
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "[fabric][ucx] handler error for topic=%s", topic
                        )

            except Exception:  # pragma: no cover
                if self._stop_event.is_set():
                    break
                logger.exception("[fabric][ucx] recv loop error")
                await asyncio.sleep(1.0)

        logger.info("[fabric][ucx] recv loop exiting")


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class EventFabric:
    """
    Facade class that selects a backend (local, zmq, ucx) based on config.

    Usage:

        fabric = EventFabric()  # mode from env
        fabric.subscribe("plan.step_ready", handler)
        fabric.publish("plan.step_ready", {"plan_id": "abc"})
        fabric.close()
    """

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        *,
        pub_endpoint: Optional[str] = None,
        sub_endpoint: Optional[str] = None,
        ucx_endpoint: Optional[str] = None,
        identity: Optional[str] = None,
    ) -> None:
        self._mode = mode.lower()
        self._backend: FabricBackend

        ident = identity or DEFAULT_IDENTITY

        if self._mode == "local":
            self._backend = LocalBackend()
        elif self._mode == "zmq":
            self._backend = ZmqBackend(
                pub_endpoint=pub_endpoint or DEFAULT_PUB_ENDPOINT,
                sub_endpoint=sub_endpoint or DEFAULT_SUB_ENDPOINT,
                identity=ident,
            )
        elif self._mode == "ucx":
            self._backend = UcxBackend(
                endpoint=ucx_endpoint or DEFAULT_UCX_ENDPOINT,
                identity=ident,
            )
        else:
            raise ValueError(f"Unknown AGI_FABRIC_MODE={mode!r}")

        logger.info("[fabric] EventFabric initialized mode=%s id=%s", self._mode, ident)

    # Public API delegates -----------------------------------------------

    def publish(self, topic: str, message: dict) -> None:
        self._backend.publish(topic, message)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        self._backend.subscribe(topic, handler)

    def close(self) -> None:
        self._backend.close()
