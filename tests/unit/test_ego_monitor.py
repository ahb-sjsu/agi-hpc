# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Ego Monitor (read-only system observer).

Verifies that the EgoMonitor correctly collects telemetry from
mocked endpoints and presents it as a structured world model.
All methods must be read-only — no writes, mutations, or controls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agi.metacognition.ego_monitor import EgoMonitor, EgoMonitorConfig


@pytest.fixture()
def monitor() -> EgoMonitor:
    """Create an EgoMonitor with default config."""
    return EgoMonitor(EgoMonitorConfig(timeout=1))


class TestEgoMonitorConfig:
    """Tests for EgoMonitorConfig defaults."""

    def test_defaults(self) -> None:
        cfg = EgoMonitorConfig()
        assert cfg.telemetry_url == "http://localhost:8081/api/telemetry"
        assert cfg.nats_monitor_url == "http://localhost:8222/varz"
        assert cfg.timeout == 5

    def test_custom_config(self) -> None:
        cfg = EgoMonitorConfig(
            telemetry_url="http://custom:9999/api/telemetry",
            timeout=2,
        )
        assert cfg.telemetry_url == "http://custom:9999/api/telemetry"
        assert cfg.timeout == 2


class TestObserve:
    """Tests for the observe() method."""

    def test_returns_dict_with_expected_keys(self, monitor) -> None:
        with patch("agi.metacognition.ego_monitor.requests.get") as mock_get:
            mock_get.side_effect = Exception("offline")
            state = monitor.observe()

        expected_keys = {
            "timestamp",
            "thermal",
            "hemispheres",
            "safety",
            "memory",
            "nats",
            "training",
            "dreaming",
            "system",
        }
        assert set(state.keys()) == expected_keys

    def test_caching(self, monitor) -> None:
        with patch("agi.metacognition.ego_monitor.requests.get") as mock_get:
            mock_get.side_effect = Exception("offline")
            state1 = monitor.observe(cache_ttl_s=60)
            state2 = monitor.observe(cache_ttl_s=60)

        # Should be the exact same object (cached)
        assert state1 is state2
        # Only one round of HTTP calls
        # First observe hits multiple endpoints; second should be cached
        first_call_count = mock_get.call_count
        _ = monitor.observe(cache_ttl_s=60)
        assert mock_get.call_count == first_call_count  # no new calls

    def test_with_telemetry_response(self, monitor) -> None:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "environment": {
                "cpu": {"max_temp": 72, "package_temps": [70, 72]},
                "gpu": [
                    {
                        "index": 0,
                        "name": "GV100",
                        "temp": 65,
                        "util": 80,
                        "mem_used": 17000,
                        "mem_total": 32768,
                    }
                ],
                "ram": {"total_gb": 224, "used_gb": 66},
            },
            "hemispheres": {
                "lh": {"status": "online", "model": "Gemma 4 31B"},
                "rh": {"status": "online", "model": "Qwen 3 32B"},
                "ego": {"status": "online", "model": "Gemma 4 E4B"},
            },
            "safety": {
                "input_checks": 42,
                "vetoes": 3,
                "avg_latency_ms": 0.8,
            },
            "memory": {
                "episodic_episodes": 150,
                "semantic_chunks": 112000,
            },
            "nats": {"status": "online"},
            "dht": {"status": "online"},
        }

        with patch(
            "agi.metacognition.ego_monitor.requests.get",
            return_value=mock_resp,
        ):
            state = monitor.observe(cache_ttl_s=0)

        assert state["thermal"]["cpu_max_temp"] == 72
        assert state["thermal"]["thermal_headroom"] == 10
        assert len(state["thermal"]["gpus"]) == 1
        assert state["hemispheres"]["lh"]["status"] == "online"
        assert state["safety"]["input_checks"] == 42


class TestSummarize:
    """Tests for the summarize() method."""

    def test_returns_string(self, monitor) -> None:
        with patch("agi.metacognition.ego_monitor.requests.get") as mock_get:
            mock_get.side_effect = Exception("offline")
            summary = monitor.summarize()

        assert isinstance(summary, str)
        assert "Atlas System State" in summary

    def test_includes_thermal_info(self, monitor) -> None:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "environment": {
                "cpu": {"max_temp": 75, "package_temps": [75]},
                "gpu": [
                    {
                        "index": 0,
                        "name": "GV100",
                        "temp": 60,
                        "util": 50,
                        "mem_used": 10000,
                        "mem_total": 32768,
                    }
                ],
                "ram": {},
            },
            "hemispheres": {},
            "safety": {},
            "memory": {},
            "nats": {},
            "dht": {},
        }

        with patch(
            "agi.metacognition.ego_monitor.requests.get",
            return_value=mock_resp,
        ):
            summary = monitor.summarize()

        assert "75°C" in summary
        assert "GPU 0" in summary


class TestReadOnlyGuarantee:
    """Tests that EgoMonitor never writes or mutates."""

    def test_no_post_requests(self, monitor) -> None:
        """EgoMonitor should only use GET requests, never POST."""
        with patch("agi.metacognition.ego_monitor.requests") as mock_req:
            mock_req.get.return_value = MagicMock(ok=False)
            monitor.observe(cache_ttl_s=0)

        # Should never call post, put, delete, patch
        mock_req.post.assert_not_called()
        mock_req.put.assert_not_called()
        mock_req.delete.assert_not_called()
        mock_req.patch.assert_not_called()

    def test_no_database_writes(self, monitor) -> None:
        """EgoMonitor should only SELECT, never INSERT/UPDATE/DELETE."""
        with (
            patch("agi.metacognition.ego_monitor.requests.get") as mock_get,
            patch("agi.metacognition.ego_monitor.psycopg2") as mock_pg,
        ):
            mock_get.side_effect = Exception("offline")
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (0,)
            mock_cursor.fetchall.return_value = []
            mock_conn.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_pg.connect.return_value = mock_conn

            monitor.observe(cache_ttl_s=0)

        # Check all SQL calls are SELECTs (read-only)
        for call in mock_cursor.execute.call_args_list:
            sql = call[0][0].strip().upper()
            assert sql.startswith(
                "SELECT"
            ), f"EgoMonitor issued non-SELECT query: {call[0][0]}"
