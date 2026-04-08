# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.core.dht.security module."""

import pytest

from agi.core.dht.security import (
    SecurityConfig,
    MTLSCredentials,
    AccessController,
    EncryptionManager,
    AuditLogger,
)


class TestSecurityConfig:
    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.enable_mtls is False
        assert cfg.enable_encryption is False
        assert cfg.encryption_algorithm == "AES-256-GCM"
        assert cfg.enable_audit is True
        assert cfg.audit_log_path == "dht_audit.log"
        assert cfg.access_control_enabled is False
        assert cfg.allowed_peers == set()

    def test_custom_values(self):
        cfg = SecurityConfig(
            enable_mtls=True,
            cert_path="/tmp/cert.pem",
            key_path="/tmp/key.pem",
            enable_encryption=True,
            access_control_enabled=True,
            allowed_peers={"peer-a", "peer-b"},
        )
        assert cfg.enable_mtls is True
        assert cfg.cert_path == "/tmp/cert.pem"
        assert cfg.key_path == "/tmp/key.pem"
        assert cfg.enable_encryption is True
        assert cfg.access_control_enabled is True
        assert cfg.allowed_peers == {"peer-a", "peer-b"}

    def test_allowed_peers_independent_instances(self):
        cfg1 = SecurityConfig()
        cfg2 = SecurityConfig()
        cfg1.allowed_peers.add("x")
        assert "x" not in cfg2.allowed_peers


class TestMTLSCredentials:
    def test_default_is_invalid(self):
        creds = MTLSCredentials()
        assert creds.cert_data == b""
        assert creds.key_data == b""
        assert creds.ca_data == b""
        assert creds.is_valid() is False

    def test_valid_when_cert_and_key_present(self):
        creds = MTLSCredentials(cert_data=b"CERT", key_data=b"KEY")
        assert creds.is_valid() is True

    def test_invalid_without_key(self):
        creds = MTLSCredentials(cert_data=b"CERT")
        assert creds.is_valid() is False

    def test_invalid_without_cert(self):
        creds = MTLSCredentials(key_data=b"KEY")
        assert creds.is_valid() is False

    def test_from_files(self, tmp_path):
        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        ca_file = tmp_path / "ca.pem"
        cert_file.write_bytes(b"CERT-DATA")
        key_file.write_bytes(b"KEY-DATA")
        ca_file.write_bytes(b"CA-DATA")

        creds = MTLSCredentials.from_files(str(cert_file), str(key_file), str(ca_file))
        assert creds.cert_data == b"CERT-DATA"
        assert creds.key_data == b"KEY-DATA"
        assert creds.ca_data == b"CA-DATA"
        assert creds.is_valid() is True

    def test_from_files_without_ca(self, tmp_path):
        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        cert_file.write_bytes(b"C")
        key_file.write_bytes(b"K")

        creds = MTLSCredentials.from_files(str(cert_file), str(key_file))
        assert creds.cert_data == b"C"
        assert creds.key_data == b"K"
        assert creds.ca_data == b""

    def test_from_files_empty_paths(self):
        creds = MTLSCredentials.from_files("", "")
        assert creds.cert_data == b""
        assert creds.key_data == b""
        assert creds.is_valid() is False


class TestAccessController:
    def test_init_default_config(self):
        ac = AccessController()
        assert ac._config is not None
        assert ac._config.access_control_enabled is False

    def test_access_allowed_when_disabled(self):
        ac = AccessController()
        assert ac.check_access("unknown-peer", "get", "some-key") is True

    def test_access_denied_unknown_peer_when_enabled(self):
        cfg = SecurityConfig(access_control_enabled=True)
        ac = AccessController(cfg)
        assert ac.check_access("unknown-peer", "get", "k") is False

    def test_allowed_peers_from_config(self):
        cfg = SecurityConfig(
            access_control_enabled=True,
            allowed_peers={"peer-1"},
        )
        ac = AccessController(cfg)
        assert ac.check_access("peer-1", "get", "k") is True
        assert ac.check_access("peer-1", "put", "k") is True
        assert ac.check_access("peer-1", "delete", "k") is True
        assert ac.check_access("peer-1", "list", "k") is True

    def test_add_peer_default_permissions(self):
        cfg = SecurityConfig(access_control_enabled=True)
        ac = AccessController(cfg)
        ac.add_peer("p1")
        assert ac.check_access("p1", "get", "k") is True
        assert ac.check_access("p1", "put", "k") is True
        assert ac.check_access("p1", "delete", "k") is True
        assert ac.check_access("p1", "list", "k") is True

    def test_add_peer_custom_permissions(self):
        cfg = SecurityConfig(access_control_enabled=True)
        ac = AccessController(cfg)
        ac.add_peer("p1", permissions=["get"])
        assert ac.check_access("p1", "get", "k") is True
        assert ac.check_access("p1", "put", "k") is False

    def test_remove_peer(self):
        cfg = SecurityConfig(access_control_enabled=True)
        ac = AccessController(cfg)
        ac.add_peer("p1")
        assert ac.check_access("p1", "get", "k") is True
        ac.remove_peer("p1")
        assert ac.check_access("p1", "get", "k") is False

    def test_get_permissions(self):
        cfg = SecurityConfig(access_control_enabled=True)
        ac = AccessController(cfg)
        ac.add_peer("p1", permissions=["get", "list"])
        perms = ac.get_permissions("p1")
        assert perms == ["get", "list"]

    def test_get_permissions_unknown_peer(self):
        ac = AccessController()
        assert ac.get_permissions("nobody") == []


class TestEncryptionManager:
    def test_init_default(self):
        em = EncryptionManager()
        assert em._config.enable_encryption is False

    def test_passthrough_when_disabled(self):
        em = EncryptionManager()
        data = b"hello world"
        assert em.encrypt(data) == data
        assert em.decrypt(data) == data

    def test_encrypt_decrypt_roundtrip(self):
        cfg = SecurityConfig(enable_encryption=True)
        em = EncryptionManager(cfg)
        data = b"sensitive data for DHT storage"
        encrypted = em.encrypt(data)
        assert encrypted != data
        decrypted = em.decrypt(encrypted)
        assert decrypted == data

    def test_encrypt_produces_different_ciphertext_each_time(self):
        cfg = SecurityConfig(enable_encryption=True)
        em = EncryptionManager(cfg)
        data = b"repeat"
        c1 = em.encrypt(data)
        c2 = em.encrypt(data)
        assert c1 != c2  # random nonce

    def test_encrypt_empty_bytes(self):
        cfg = SecurityConfig(enable_encryption=True)
        em = EncryptionManager(cfg)
        encrypted = em.encrypt(b"")
        decrypted = em.decrypt(encrypted)
        assert decrypted == b""

    def test_decrypt_too_short_raises(self):
        cfg = SecurityConfig(enable_encryption=True)
        em = EncryptionManager(cfg)
        with pytest.raises(ValueError, match="too short"):
            em.decrypt(b"short")

    def test_key_derived_from_cert_path(self):
        cfg1 = SecurityConfig(enable_encryption=True, cert_path="/path/a")
        cfg2 = SecurityConfig(enable_encryption=True, cert_path="/path/b")
        em1 = EncryptionManager(cfg1)
        em2 = EncryptionManager(cfg2)
        assert em1._key != em2._key


class TestAuditLogger:
    def test_init_default(self):
        al = AuditLogger()
        assert al._config.enable_audit is True
        assert al.get_recent_entries() == []

    def test_log_access(self):
        al = AuditLogger()
        al.log_access("peer-1", "get", "my-key", True)
        entries = al.get_recent_entries()
        assert len(entries) == 1
        assert entries[0]["type"] == "access"
        assert entries[0]["peer_id"] == "peer-1"
        assert entries[0]["operation"] == "get"
        assert entries[0]["key"] == "my-key"
        assert entries[0]["allowed"] is True
        assert "timestamp" in entries[0]

    def test_log_error(self):
        al = AuditLogger()
        al.log_error("peer-2", "connection refused")
        entries = al.get_recent_entries()
        assert len(entries) == 1
        assert entries[0]["type"] == "error"
        assert entries[0]["peer_id"] == "peer-2"
        assert entries[0]["error"] == "connection refused"

    def test_recent_entries_newest_first(self):
        al = AuditLogger()
        al.log_access("p1", "get", "k1", True)
        al.log_access("p2", "put", "k2", False)
        al.log_error("p3", "timeout")
        entries = al.get_recent_entries()
        assert len(entries) == 3
        assert entries[0]["peer_id"] == "p3"
        assert entries[2]["peer_id"] == "p1"

    def test_recent_entries_with_count(self):
        al = AuditLogger()
        for i in range(10):
            al.log_access(f"p{i}", "get", "k", True)
        entries = al.get_recent_entries(count=3)
        assert len(entries) == 3

    def test_audit_disabled_skips_logging(self):
        cfg = SecurityConfig(enable_audit=False)
        al = AuditLogger(cfg)
        al.log_access("peer-1", "get", "key", True)
        al.log_error("peer-1", "error")
        assert al.get_recent_entries() == []

    def test_mixed_access_and_error_entries(self):
        al = AuditLogger()
        al.log_access("p1", "put", "k1", True)
        al.log_error("p2", "bad request")
        al.log_access("p3", "delete", "k2", False)
        entries = al.get_recent_entries()
        assert len(entries) == 3
        types = [e["type"] for e in entries]
        assert "access" in types
        assert "error" in types
