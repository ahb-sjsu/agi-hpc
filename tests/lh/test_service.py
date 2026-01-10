"""Tests for LH service module."""

import pytest
import sys
from unittest.mock import MagicMock, patch

from agi.lh.service import LHServiceConfig, LHService, parse_args, main


class TestLHServiceConfig:
    """Tests for LHServiceConfig dataclass."""

    def test_default_config_values(self):
        """LHServiceConfig should have sensible defaults."""
        config = LHServiceConfig()

        assert config.port == 50100
        assert config.max_workers == 16
        assert config.fabric_mode == "local"
        assert config.fabric_identity == "LH"

    def test_config_accepts_custom_values(self):
        """LHServiceConfig should accept custom values."""
        config = LHServiceConfig(
            port=60000,
            max_workers=8,
            fabric_mode="zmq",
            fabric_identity="LH-test",
        )

        assert config.port == 60000
        assert config.max_workers == 8
        assert config.fabric_mode == "zmq"
        assert config.fabric_identity == "LH-test"

    def test_config_service_addresses(self):
        """LHServiceConfig should have service address fields."""
        config = LHServiceConfig(
            memory_addr="memory:50110",
            safety_addr="safety:50120",
            meta_addr="meta:50130",
        )

        assert config.memory_addr == "memory:50110"
        assert config.safety_addr == "safety:50120"
        assert config.meta_addr == "meta:50130"


class TestLHService:
    """Tests for LHService class."""

    def test_service_instantiation(self):
        """LHService should instantiate with config."""
        config = LHServiceConfig(port=50199)
        service = LHService(config)

        assert service._cfg.port == 50199

    def test_service_default_config(self):
        """LHService should use default config if none provided."""
        service = LHService()

        assert service._cfg.port == 50100

    def test_service_stores_config(self):
        """LHService should store the provided config."""
        config = LHServiceConfig(
            port=50200,
            max_workers=4,
            memory_addr="mem:1234",
            safety_addr="safe:1234",
            meta_addr="meta:1234",
        )
        service = LHService(config)

        assert service._cfg is config

    def test_service_initial_state(self):
        """LHService should start with no server or fabric."""
        service = LHService()

        assert service._server is None
        assert service._fabric is None


class TestLHServiceLifecycle:
    """Tests for LHService start/stop lifecycle."""

    def test_stop_before_start_is_safe(self):
        """Calling stop() before start() should not raise."""
        service = LHService()
        # Should not raise even though service wasn't started
        service.stop()

    def test_service_config_from_environment(self):
        """LHServiceConfig should read from environment variables."""
        import os

        # Test that config reads AGI_LH_PORT if set
        original = os.environ.get("AGI_LH_PORT")
        try:
            os.environ["AGI_LH_PORT"] = "55555"
            config = LHServiceConfig()
            assert config.port == 55555
        finally:
            if original:
                os.environ["AGI_LH_PORT"] = original
            else:
                os.environ.pop("AGI_LH_PORT", None)


class TestParseArgs:
    """Tests for command line argument parsing."""

    def test_parse_args_defaults(self):
        """parse_args should return defaults when no args provided."""
        with patch.object(sys, "argv", ["service.py"]):
            args = parse_args()

        assert args.port is None
        assert args.fabric_mode is None
        assert args.no_safety is False
        assert args.no_meta is False
        assert args.debug is False

    def test_parse_args_with_port(self):
        """parse_args should accept --port argument."""
        with patch.object(sys, "argv", ["service.py", "--port", "55000"]):
            args = parse_args()

        assert args.port == 55000

    def test_parse_args_with_fabric_mode(self):
        """parse_args should accept --fabric-mode argument."""
        with patch.object(sys, "argv", ["service.py", "--fabric-mode", "zmq"]):
            args = parse_args()

        assert args.fabric_mode == "zmq"

    def test_parse_args_with_flags(self):
        """parse_args should accept flag arguments."""
        with patch.object(
            sys, "argv", ["service.py", "--no-safety", "--no-meta", "--debug"]
        ):
            args = parse_args()

        assert args.no_safety is True
        assert args.no_meta is True
        assert args.debug is True


class TestLHServiceMethods:
    """Tests for LHService instance methods."""

    def test_wait_with_no_server(self):
        """wait() should be safe when no server exists."""
        service = LHService()
        # Should not raise
        service.wait()

    def test_stop_closes_fabric(self):
        """stop() should close fabric if it exists."""
        service = LHService()
        mock_fabric = MagicMock()
        service._fabric = mock_fabric

        service.stop()

        mock_fabric.close.assert_called_once()

    def test_stop_stops_server(self):
        """stop() should stop server if it exists."""
        service = LHService()
        mock_server = MagicMock()
        service._server = mock_server

        service.stop()

        mock_server.stop.assert_called_once_with(grace=5.0)

    def test_wait_with_server(self):
        """wait() should call server.wait() when server exists."""
        service = LHService()
        mock_server = MagicMock()
        service._server = mock_server

        service.wait()

        mock_server.wait.assert_called_once()

    def test_stop_handles_both_server_and_fabric(self):
        """stop() should handle both server and fabric."""
        service = LHService()
        mock_server = MagicMock()
        mock_fabric = MagicMock()
        service._server = mock_server
        service._fabric = mock_fabric

        service.stop()

        mock_server.stop.assert_called_once()
        mock_fabric.close.assert_called_once()


class TestMain:
    """Tests for main() entry point."""

    def test_main_creates_service_and_runs(self):
        """main() should create LHService and start it."""
        with patch.object(sys, "argv", ["service.py"]):
            with patch("agi.lh.service.LHService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service
                # Make wait() raise KeyboardInterrupt to exit
                mock_service.wait.side_effect = KeyboardInterrupt()

                result = main()

                mock_service_class.assert_called_once()
                mock_service.start.assert_called_once()
                mock_service.stop.assert_called_once()
                assert result == 0

    def test_main_applies_port_arg(self):
        """main() should apply --port to config."""
        with patch.object(sys, "argv", ["service.py", "--port", "60000"]):
            with patch("agi.lh.service.LHService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service
                mock_service.wait.side_effect = KeyboardInterrupt()

                main()

                # Check the config passed to LHService
                call_kwargs = mock_service_class.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
                assert config.port == 60000

    def test_main_applies_disable_flags(self):
        """main() should apply --no-safety and --no-meta flags."""
        with patch.object(sys, "argv", ["service.py", "--no-safety", "--no-meta"]):
            with patch("agi.lh.service.LHService") as mock_service_class:
                mock_service = MagicMock()
                mock_service_class.return_value = mock_service
                mock_service.wait.side_effect = KeyboardInterrupt()

                main()

                call_kwargs = mock_service_class.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
                assert config.enable_safety is False
                assert config.enable_metacognition is False
