# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Redis Streams backend for Event Fabric.

Provides persistent messaging with:
- At-least-once delivery via consumer groups
- Message replay from any point in time
- Dead letter queue for failed messages
- Automatic stream trimming

Environment Variables:
    AGI_FABRIC_REDIS_URL       Redis URL (default: redis://localhost:6379)
    AGI_FABRIC_REDIS_PREFIX    Stream key prefix (default: fabric:)
    AGI_FABRIC_CONSUMER_GROUP  Consumer group name (default: agi-hpc)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    import redis
    import redis.asyncio as aioredis
except ImportError:
    redis = None  # type: ignore
    aioredis = None  # type: ignore

# Configuration defaults
DEFAULT_REDIS_URL = os.getenv("AGI_FABRIC_REDIS_URL", "redis://localhost:6379")
DEFAULT_PREFIX = os.getenv("AGI_FABRIC_REDIS_PREFIX", "fabric:")
DEFAULT_CONSUMER_GROUP = os.getenv("AGI_FABRIC_CONSUMER_GROUP", "agi-hpc")

EventHandler = Callable[[dict], None]


@dataclass
class RedisBackendConfig:
    """Configuration for Redis backend."""

    url: str = DEFAULT_REDIS_URL
    stream_prefix: str = DEFAULT_PREFIX
    consumer_group: str = DEFAULT_CONSUMER_GROUP
    consumer_name: Optional[str] = None
    max_stream_length: int = 100000
    block_timeout_ms: int = 1000
    batch_size: int = 100
    retry_count: int = 3
    dead_letter_suffix: str = ":dlq"
    enable_dlq: bool = True


@dataclass
class StreamMessage:
    """A message from a Redis stream."""

    message_id: str
    topic: str
    data: Dict[str, Any]
    timestamp: float = 0.0
    retry_count: int = 0

    @classmethod
    def from_stream_entry(
        cls,
        message_id: str,
        fields: Dict[bytes, bytes],
        topic: str,
    ) -> "StreamMessage":
        """Create from Redis stream entry."""
        data_raw = fields.get(b"data", b"{}")
        try:
            data = json.loads(data_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}

        ts_raw = fields.get(b"timestamp", b"0")
        try:
            timestamp = float(ts_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            timestamp = 0.0

        retry_raw = fields.get(b"retry_count", b"0")
        try:
            retry_count = int(retry_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            retry_count = 0

        return cls(
            message_id=(
                message_id if isinstance(message_id, str) else message_id.decode()
            ),
            topic=topic,
            data=data,
            timestamp=timestamp,
            retry_count=retry_count,
        )


class RedisBackend:
    """
    Redis Streams backend for Event Fabric.

    Features:
    - Persistent message storage
    - Consumer groups for load balancing
    - Message acknowledgment
    - Replay from any point
    - Dead letter queue for failed messages
    """

    def __init__(self, config: Optional[RedisBackendConfig] = None) -> None:
        """Initialize Redis backend."""
        if redis is None:
            raise RuntimeError("redis-py is required but not installed")

        self.config = config or RedisBackendConfig()
        self.config.consumer_name = (
            self.config.consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        )

        self._client: Optional[redis.Redis] = None
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None
        self._connected = False

        logger.info(
            "[fabric][redis] initialized url=%s prefix=%s group=%s consumer=%s",
            self.config.url,
            self.config.stream_prefix,
            self.config.consumer_group,
            self.config.consumer_name,
        )

    def connect(self) -> None:
        """Connect to Redis."""
        self._client = redis.from_url(
            self.config.url,
            decode_responses=False,  # We handle decoding ourselves
        )
        # Test connection
        self._client.ping()
        self._connected = True
        logger.info("[fabric][redis] connected to %s", self.config.url)

    def _stream_key(self, topic: str) -> str:
        """Get Redis stream key for topic."""
        return f"{self.config.stream_prefix}{topic}"

    def _dlq_key(self, topic: str) -> str:
        """Get dead letter queue key for topic."""
        return f"{self.config.stream_prefix}{topic}{self.config.dead_letter_suffix}"

    def publish(self, topic: str, message: dict) -> str:
        """Publish message to Redis Stream.

        Args:
            topic: Topic name
            message: Message payload

        Returns:
            Message ID from Redis
        """
        if not self._connected:
            self.connect()

        stream_key = self._stream_key(topic)
        payload = json.dumps(message)

        entry = {
            "data": payload,
            "timestamp": str(time.time()),
            "retry_count": "0",
        }

        message_id = self._client.xadd(
            stream_key,
            entry,
            maxlen=self.config.max_stream_length,
            approximate=True,
        )

        logger.debug(
            "[fabric][redis] published topic=%s id=%s",
            topic,
            message_id.decode() if isinstance(message_id, bytes) else message_id,
        )

        return message_id.decode() if isinstance(message_id, bytes) else message_id

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to topic with consumer group.

        Args:
            topic: Topic name
            handler: Callback function for messages
        """
        if not self._connected:
            self.connect()

        stream_key = self._stream_key(topic)

        # Create consumer group if not exists
        try:
            self._client.xgroup_create(
                stream_key,
                self.config.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "[fabric][redis] created consumer group %s for %s",
                self.config.consumer_group,
                stream_key,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

        # Start consumer thread if not running
        if self._recv_thread is None or not self._recv_thread.is_alive():
            self._recv_thread = threading.Thread(
                target=self._consume_loop,
                name=f"fabric-redis-{self.config.consumer_name}",
                daemon=True,
            )
            self._recv_thread.start()

        logger.info("[fabric][redis] subscribed topic=%s", topic)

    def unsubscribe(self, topic: str, handler: Optional[EventHandler] = None) -> None:
        """Unsubscribe from topic.

        Args:
            topic: Topic name
            handler: Specific handler to remove, or None to remove all
        """
        with self._lock:
            if topic not in self._subscribers:
                return

            if handler is None:
                del self._subscribers[topic]
            else:
                self._subscribers[topic] = [
                    h for h in self._subscribers[topic] if h != handler
                ]
                if not self._subscribers[topic]:
                    del self._subscribers[topic]

        logger.info("[fabric][redis] unsubscribed topic=%s", topic)

    def replay(
        self,
        topic: str,
        start_id: str = "0",
        end_id: str = "+",
        count: int = 100,
    ) -> List[StreamMessage]:
        """Replay messages from stream history.

        Args:
            topic: Topic name
            start_id: Start message ID (inclusive)
            end_id: End message ID (inclusive), "+" for latest
            count: Maximum messages to return

        Returns:
            List of messages
        """
        if not self._connected:
            self.connect()

        stream_key = self._stream_key(topic)
        messages = self._client.xrange(stream_key, start_id, end_id, count=count)

        result = []
        for message_id, fields in messages:
            msg = StreamMessage.from_stream_entry(message_id, fields, topic)
            result.append(msg)

        logger.debug(
            "[fabric][redis] replayed %d messages from %s",
            len(result),
            topic,
        )

        return result

    def get_stream_info(self, topic: str) -> Dict[str, Any]:
        """Get stream information.

        Args:
            topic: Topic name

        Returns:
            Stream info dict
        """
        if not self._connected:
            self.connect()

        stream_key = self._stream_key(topic)

        try:
            info = self._client.xinfo_stream(stream_key)
            return {
                "length": info.get(b"length", 0),
                "first_entry": info.get(b"first-entry"),
                "last_entry": info.get(b"last-entry"),
                "groups": info.get(b"groups", 0),
            }
        except redis.ResponseError:
            return {"length": 0, "error": "stream not found"}

    def get_pending(self, topic: str) -> List[Dict]:
        """Get pending messages for consumer group.

        Args:
            topic: Topic name

        Returns:
            List of pending message info
        """
        if not self._connected:
            self.connect()

        stream_key = self._stream_key(topic)

        try:
            pending = self._client.xpending_range(
                stream_key,
                self.config.consumer_group,
                min="-",
                max="+",
                count=100,
            )
            return [
                {
                    "message_id": (
                        p["message_id"].decode()
                        if isinstance(p["message_id"], bytes)
                        else p["message_id"]
                    ),
                    "consumer": (
                        p["consumer"].decode()
                        if isinstance(p["consumer"], bytes)
                        else p["consumer"]
                    ),
                    "idle_time": p["time_since_delivered"],
                    "delivery_count": p["times_delivered"],
                }
                for p in pending
            ]
        except redis.ResponseError:
            return []

    def _consume_loop(self) -> None:
        """Consumer loop reading from all subscribed streams."""
        logger.info(
            "[fabric][redis] consumer loop started consumer=%s",
            self.config.consumer_name,
        )

        while not self._stop_event.is_set():
            with self._lock:
                topics = list(self._subscribers.keys())

            if not topics:
                self._stop_event.wait(1.0)
                continue

            # Build streams dict for XREADGROUP
            streams = {self._stream_key(t): ">" for t in topics}

            try:
                messages = self._client.xreadgroup(
                    self.config.consumer_group,
                    self.config.consumer_name,
                    streams,
                    count=self.config.batch_size,
                    block=self.config.block_timeout_ms,
                )
            except redis.ConnectionError:
                logger.warning("[fabric][redis] connection lost, reconnecting")
                try:
                    self.connect()
                except Exception:
                    self._stop_event.wait(1.0)
                continue
            except Exception:
                logger.exception("[fabric][redis] xreadgroup error")
                self._stop_event.wait(1.0)
                continue

            if not messages:
                continue

            for stream_key, stream_messages in messages:
                topic = (
                    stream_key.decode() if isinstance(stream_key, bytes) else stream_key
                )
                topic = topic.replace(self.config.stream_prefix, "")
                self._process_messages(topic, stream_messages)

        logger.info("[fabric][redis] consumer loop exiting")

    def _process_messages(
        self,
        topic: str,
        messages: List[tuple],
    ) -> None:
        """Process messages and acknowledge."""
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))

        stream_key = self._stream_key(topic)

        for message_id, fields in messages:
            msg = StreamMessage.from_stream_entry(message_id, fields, topic)

            success = True
            for handler in handlers:
                try:
                    handler(msg.data)
                except Exception:
                    logger.exception(
                        "[fabric][redis] handler error topic=%s id=%s",
                        topic,
                        msg.message_id,
                    )
                    success = False

            if success:
                # Acknowledge successful processing
                self._client.xack(
                    stream_key,
                    self.config.consumer_group,
                    msg.message_id,
                )
            else:
                # Handle failure
                self._handle_failed_message(topic, msg)

    def _handle_failed_message(self, topic: str, msg: StreamMessage) -> None:
        """Handle a message that failed processing."""
        stream_key = self._stream_key(topic)

        if msg.retry_count >= self.config.retry_count:
            # Move to dead letter queue
            if self.config.enable_dlq:
                dlq_key = self._dlq_key(topic)
                self._client.xadd(
                    dlq_key,
                    {
                        "data": json.dumps(msg.data),
                        "original_id": msg.message_id,
                        "timestamp": str(time.time()),
                        "error": "max_retries_exceeded",
                    },
                )
                logger.warning(
                    "[fabric][redis] message moved to DLQ topic=%s id=%s",
                    topic,
                    msg.message_id,
                )

            # Acknowledge to remove from pending
            self._client.xack(
                stream_key,
                self.config.consumer_group,
                msg.message_id,
            )
        else:
            # Will be retried on next claim
            logger.info(
                "[fabric][redis] message will retry topic=%s id=%s count=%d",
                topic,
                msg.message_id,
                msg.retry_count + 1,
            )

    def close(self) -> None:
        """Close the backend."""
        self._stop_event.set()

        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)

        if self._client:
            self._client.close()
            self._client = None

        self._connected = False
        logger.info("[fabric][redis] closed")


class AsyncRedisBackend:
    """Async version of Redis backend using redis.asyncio."""

    def __init__(self, config: Optional[RedisBackendConfig] = None) -> None:
        """Initialize async Redis backend."""
        if aioredis is None:
            raise RuntimeError("redis-py with asyncio support is required")

        self.config = config or RedisBackendConfig()
        self.config.consumer_name = (
            self.config.consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        )

        self._client: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        self._client = await aioredis.from_url(
            self.config.url,
            decode_responses=False,
        )
        await self._client.ping()
        self._connected = True
        logger.info("[fabric][redis][async] connected to %s", self.config.url)

    def _stream_key(self, topic: str) -> str:
        return f"{self.config.stream_prefix}{topic}"

    async def publish(self, topic: str, message: dict) -> str:
        """Publish message asynchronously."""
        if not self._connected:
            await self.connect()

        stream_key = self._stream_key(topic)
        payload = json.dumps(message)

        entry = {
            "data": payload,
            "timestamp": str(time.time()),
        }

        message_id = await self._client.xadd(
            stream_key,
            entry,
            maxlen=self.config.max_stream_length,
            approximate=True,
        )

        return message_id.decode() if isinstance(message_id, bytes) else message_id

    async def replay(
        self,
        topic: str,
        start_id: str = "0",
        end_id: str = "+",
        count: int = 100,
    ) -> List[StreamMessage]:
        """Replay messages asynchronously."""
        if not self._connected:
            await self.connect()

        stream_key = self._stream_key(topic)
        messages = await self._client.xrange(stream_key, start_id, end_id, count=count)

        return [
            StreamMessage.from_stream_entry(mid, fields, topic)
            for mid, fields in messages
        ]

    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.close()
            self._client = None
        self._connected = False
        logger.info("[fabric][redis][async] closed")


# ---------------------------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------------------------


def export_stream_to_jsonl(
    topic: str,
    output_path: str,
    start_id: str = "0",
    end_id: str = "+",
    batch_size: int = 1000,
    config: Optional[RedisBackendConfig] = None,
) -> int:
    """Export a Redis stream to JSONL file.

    Args:
        topic: Topic name to export
        output_path: Output file path
        start_id: Start message ID
        end_id: End message ID
        batch_size: Batch size for reading
        config: Redis backend configuration

    Returns:
        Number of messages exported
    """
    backend = RedisBackend(config)
    backend.connect()

    total_exported = 0
    current_id = start_id

    try:
        with open(output_path, "w") as f:
            while True:
                messages = backend.replay(
                    topic,
                    start_id=current_id,
                    end_id=end_id,
                    count=batch_size,
                )

                if not messages:
                    break

                for msg in messages:
                    record = {
                        "message_id": msg.message_id,
                        "topic": msg.topic,
                        "timestamp": msg.timestamp,
                        "data": msg.data,
                    }
                    f.write(json.dumps(record) + "\n")
                    total_exported += 1

                # Move to next batch (exclusive of last ID)
                last_id = messages[-1].message_id
                # Increment the sequence number for exclusive range
                parts = last_id.split("-")
                if len(parts) == 2:
                    current_id = f"{parts[0]}-{int(parts[1]) + 1}"
                else:
                    break

                if len(messages) < batch_size:
                    break

        logger.info(
            "[fabric][redis] exported %d messages from %s to %s",
            total_exported,
            topic,
            output_path,
        )

    finally:
        backend.close()

    return total_exported


def import_stream_from_jsonl(
    input_path: str,
    topic: Optional[str] = None,
    config: Optional[RedisBackendConfig] = None,
) -> int:
    """Import messages from JSONL file to Redis stream.

    Args:
        input_path: Input JSONL file path
        topic: Override topic (uses original if None)
        config: Redis backend configuration

    Returns:
        Number of messages imported
    """
    backend = RedisBackend(config)
    backend.connect()

    total_imported = 0

    try:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    target_topic = topic or record.get("topic", "unknown")
                    data = record.get("data", {})
                    backend.publish(target_topic, data)
                    total_imported += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("[fabric][redis] skip invalid record: %s", e)
                    continue

        logger.info(
            "[fabric][redis] imported %d messages from %s",
            total_imported,
            input_path,
        )

    finally:
        backend.close()

    return total_imported


def get_stream_stats(
    topics: List[str],
    config: Optional[RedisBackendConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Get statistics for multiple streams.

    Args:
        topics: List of topic names
        config: Redis backend configuration

    Returns:
        Dict mapping topic to stats
    """
    backend = RedisBackend(config)
    backend.connect()

    stats = {}

    try:
        for topic in topics:
            info = backend.get_stream_info(topic)
            pending = backend.get_pending(topic)
            stats[topic] = {
                **info,
                "pending_count": len(pending),
                "pending_messages": pending[:10],  # First 10 only
            }
    finally:
        backend.close()

    return stats


def cleanup_streams(
    topics: List[str],
    max_age_seconds: float = 86400 * 7,  # 7 days
    config: Optional[RedisBackendConfig] = None,
) -> Dict[str, int]:
    """Clean up old messages from streams.

    Args:
        topics: List of topic names
        max_age_seconds: Maximum age for messages
        config: Redis backend configuration

    Returns:
        Dict mapping topic to number of messages deleted
    """
    if redis is None:
        raise RuntimeError("redis-py is required")

    config = config or RedisBackendConfig()
    client = redis.from_url(config.url, decode_responses=False)

    deleted = {}
    cutoff_time = int((time.time() - max_age_seconds) * 1000)
    cutoff_id = f"{cutoff_time}-0"

    try:
        for topic in topics:
            stream_key = f"{config.stream_prefix}{topic}"
            try:
                count = client.xtrim(stream_key, minid=cutoff_id)
                deleted[topic] = count
                logger.info(
                    "[fabric][redis] cleaned %d messages from %s",
                    count,
                    topic,
                )
            except redis.ResponseError as e:
                logger.warning("[fabric][redis] cleanup failed for %s: %s", topic, e)
                deleted[topic] = 0
    finally:
        client.close()

    return deleted
