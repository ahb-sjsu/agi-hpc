"""
Event fabric stub.

This is a simple in-process pub/sub dispatcher.
Later replace with UCX/ZeroMQ-based transport for inter-node communication.
"""

from typing import Callable, Dict, List
import threading


class EventFabric:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, message: dict):
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for fn in handlers:
            fn(message)

    def subscribe(self, topic: str, handler: Callable):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        print(f"[event-fabric] Subscribed to {topic} -> {handler.__name__}")
