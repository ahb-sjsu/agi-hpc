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
Event fabric abstraction for AGI-HPC.

Backends:
    - local : in-process pub/sub (no networking, ideal for tests)
    - zmq   : ZeroMQ PUB/SUB (simple multi-node)
    - ucx   : UCX via ucx-py (HPC-grade transport: RDMA, SHM, TCP fallback)

Mode selection via environment variable:

    AGI_FABRIC_MODE = "local" | "zmq" | "ucx"

Additional configuration:

    # ZMQ backend
    AGI_FABRIC_PUB_ENDPOINT = "tcp://fabric:5556"
    AGI_FABRIC_SUB_ENDPOINT = "tcp://fabric:5555"

    # UCX backend
    AGI_FABRIC_UCX_ENDPOINT = "tcp://fabric:13337"

Public API:

    fabric = EventFabric()
    fabric.subscribe("topic", handler)
    fabric.publish("topic", {"data": 1})
    fabric.close()
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

try:  # pyzmq optional
    import zmq  # type: ignore
except Exception:
    zmq = None  # type: ignore

try:  # ucx-py optional
    import ucp  # type: ignore
except Exception:
    ucp = None  # type: ignore


DEFAULT_MODE = os.getenv("AGI_FABRIC_MODE", "local").lower()
DEFAULT_PUB_ENDPOINT = os.getenv("AGI_FABRIC_PUB_ENDPOINT", "tcp://fabric:5556")
DEFAULT_SUB_ENDPOINT = os.getenv("AGI_FABRIC_SUB_ENDPOINT", "tcp://fabric:5555")
DEFAULT_UCX_ENDPOINT = os.getenv("AGI_FABRIC_UCX_ENDPOINT", "tcp://fabric:13337")
DEFAULT_IDENTITY = os.getenv("AGI_FABRIC_IDENTITY", "node")

EventHandler = Callable[[dict], None]


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FabricBackend(Protocol):
    def publish(self, topic: str, message: dict) -> None: ...

    def subscribe(self, topic: str, handler: EventHandler) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# LOCAL BACKEND
# ---------------------------------------------------------------------------


class LocalBackend:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        logger.info("[fabric][local] initialized")

    def publish(self, topic: str, message: dict) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for fn in handlers:
            try:
                fn(message)
            except Exception:
                logger.exception("[fabric][local] handler error topic=%s", topic)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        logger.info("[fabric][local] subscribed topic=%s", topic)

    def close(self) -> None:
        with self._lock:
            self._subscribers.clear()
        logger.info("[fabric][local] closed")


# ---------------------------------------------------------------------------
# ZEROMQ BACKEND
# ---------------------------------------------------------------------------


class ZmqBackend:
    """
    ZeroMQ PUB/SUB with a broker providing XPUB/XSUB.
    """

    def __init__(
        self,
        pub_endpoint: str = DEFAULT_PUB_ENDPOINT,
        sub_endpoint: str = DEFAULT_SUB_ENDPOINT,
        identity: str = DEFAULT_IDENTITY,
    ) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq not installed")

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._sub = self._ctx.socket(zmq.SUB)

        self._identity = identity
        self._pub.setsockopt_string(zmq.IDENTITY, identity)
        self._sub.setsockopt_string(zmq.IDENTITY, identity)

        self._pub.connect(pub_endpoint)
        self._sub.connect(sub_endpoint)

        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

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

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def publish(self, topic: str, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        try:
            self._pub.send_multipart([topic.encode("utf-8"), payload])
        except Exception:
            logger.exception("[fabric][zmq] publish failed topic=%s", topic)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info("[fabric][zmq] subscribed topic=%s", topic)

    def close(self) -> None:
        self._stop_event.set()
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        try:
            self._sub.close(linger=0)
            self._pub.close(linger=0)
        except Exception:
            logger.exception("[fabric][zmq] close error")
        logger.info("[fabric][zmq] closed")

    # ---------------------------------------------------------
    # ZMQ recv loop (refactored to avoid C901)
    # ---------------------------------------------------------

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        logger.info("[fabric][zmq] recv loop started id=%s", self._identity)

        while not self._stop_event.is_set():
            if not self._poll_ready(poller):
                continue
            frames = self._recv_frames()
            if frames is None:
                continue
            self._handle_frames(frames)

        logger.info("[fabric][zmq] recv loop exiting id=%s", self._identity)

    def _poll_ready(self, poller) -> bool:
        try:
            events = dict(poller.poll(timeout=1000))
        except Exception:
            if not self._stop_event.is_set():
                logger.exception("[fabric][zmq] poll error")
            return False
        return self._sub in events and events[self._sub] & zmq.POLLIN

    def _recv_frames(self) -> Optional[List[bytes]]:
        try:
            return self._sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except Exception:
            logger.exception("[fabric][zmq] recv error")
            return None

    def _handle_frames(self, frames: List[bytes]) -> None:
        if len(frames) != 2:
            logger.warning("[fabric][zmq] malformed frames=%d", len(frames))
            return

        topic_bytes, data = frames
        topic = topic_bytes.decode("utf-8", errors="ignore")

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            logger.exception("[fabric][zmq] JSON decode failed topic=%s", topic)
            return

        with self._lock:
            handlers = list(self._subscribers.get(topic, []))

        for fn in handlers:
            try:
                fn(msg)
            except Exception:
                logger.exception("[fabric][zmq] handler error topic=%s", topic)


# ---------------------------------------------------------------------------
# UCX BACKEND
# ---------------------------------------------------------------------------


@dataclass
class _UcxState:
    endpoint_str: str
    ep: Optional["ucp.Endpoint"] = None  # type: ignore[name-defined]
    connected: bool = False


class UcxBackend:
    """
    High-performance UCX backend.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_UCX_ENDPOINT,
        identity: str = DEFAULT_IDENTITY,
    ) -> None:
        if ucp is None:
            raise RuntimeError("ucx-py not installed")

        self._identity = identity
        self._state = _UcxState(endpoint_str=endpoint)

        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Dedicated event loop thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"fabric-ucx-loop-{identity}",
            daemon=True,
        )
        self._thread.start()

        self._loop.call_soon_threadsafe(ucp.init)
        asyncio.run_coroutine_threadsafe(self._connect_and_loop(), self._loop)

        logger.info(
            "[fabric][ucx] initialized id=%s endpoint=%s",
            identity,
            endpoint,
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def publish(self, topic: str, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        wire = topic.encode("utf-8") + b"\0" + payload
        size_prefix = len(wire).to_bytes(4, "little")
        data = size_prefix + wire

        async def _send() -> None:
            if not self._state.connected or self._state.ep is None:
                return
            try:
                await self._state.ep.send(data)
            except Exception:
                logger.exception("[fabric][ucx] send failed topic=%s", topic)

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        logger.info("[fabric][ucx] subscribed topic=%s", topic)

    def close(self) -> None:
        self._stop_event.set()

        async def _shutdown() -> None:
            try:
                if self._state.ep is not None:
                    await self._state.ep.close()
            except Exception:
                logger.exception("[fabric][ucx] error closing endpoint")

            try:
                ucp.finalize()
            except Exception:
                logger.exception("[fabric][ucx] finalize error")

            self._loop.stop()

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            try:
                fut.result(timeout=2.0)
            except Exception:
                pass

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info("[fabric][ucx] closed id=%s", self._identity)

    # ---------------------------------------------------------
    # UCX loop logic (C901-friendly helpers)
    # ---------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_and_loop(self) -> None:
        host, port = self._parse_ucx_addr(self._state.endpoint_str)

        try:
            ep = await ucp.create_endpoint(host, port)
        except Exception:
            logger.exception("[fabric][ucx] failed to connect")
            return

        self._state.ep = ep
        self._state.connected = True
        logger.info("[fabric][ucx] connected to %s:%d", host, port)

        while not self._stop_event.is_set():
            payload = await self._recv_one(ep)
            if payload is None:
                continue
            self._dispatch_ucx(payload)

        logger.info("[fabric][ucx] recv loop exiting")

    def _parse_ucx_addr(self, addr: str) -> tuple[str, int]:
        if not addr.startswith("tcp://"):
            logger.warning("[fabric][ucx] only tcp:// supported: %s", addr)
        host_port = addr.replace("tcp://", "")
        host, port_s = host_port.split(":")
        return host, int(port_s)

    async def _recv_one(self, ep) -> Optional[bytes]:
        try:
            hdr = bytearray(4)
            await ep.recv(hdr)
            size = int.from_bytes(hdr, "little")

            if size <= 0 or size > 16 * 1024 * 1024:
                logger.warning("[fabric][ucx] invalid size=%d", size)
                return None

            payload = bytearray(size)
            await ep.recv(payload)
            return payload
        except Exception:
            if not self._stop_event.is_set():
                logger.exception("[fabric][ucx] recv error")
            return None

    def _dispatch_ucx(self, payload: bytes) -> None:
        try:
            topic_bytes, data = payload.split(b"\0", 1)
        except ValueError:
            logger.warning("[fabric][ucx] missing separator")
            return

        topic = topic_bytes.decode("utf-8", errors="ignore")

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            logger.exception("[fabric][ucx] JSON decode failed topic=%s", topic)
            return

        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for fn in handlers:
            try:
                fn(msg)
            except Exception:
                logger.exception("[fabric][ucx] handler error topic=%s", topic)


# ---------------------------------------------------------------------------
# PUBLIC FACADE
# ---------------------------------------------------------------------------


class EventFabric:
    """
    High-level API selecting backend: local, zmq, or ucx.
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
        mode = mode.lower()
        identity = identity or DEFAULT_IDENTITY

        if mode == "local":
            backend: FabricBackend = LocalBackend()
        elif mode == "zmq":
            backend = ZmqBackend(
                pub_endpoint=pub_endpoint or DEFAULT_PUB_ENDPOINT,
                sub_endpoint=sub_endpoint or DEFAULT_SUB_ENDPOINT,
                identity=identity,
            )
        elif mode == "ucx":
            backend = UcxBackend(
                endpoint=ucx_endpoint or DEFAULT_UCX_ENDPOINT,
                identity=identity,
            )
        else:
            raise ValueError(f"Unknown AGI_FABRIC_MODE={mode!r}")

        self._backend = backend
        logger.info("[fabric] initialized mode=%s id=%s", mode, identity)

    def publish(self, topic: str, message: dict) -> None:
        self._backend.publish(topic, message)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        self._backend.subscribe(topic, handler)

    def close(self) -> None:
        self._backend.close()
