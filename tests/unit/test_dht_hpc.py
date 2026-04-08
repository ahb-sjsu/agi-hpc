# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.core.dht.hpc module."""

import os
from unittest.mock import patch

from agi.core.dht.hpc import HPCConfig


class TestHPCConfig:
    def test_default(self):
        cfg = HPCConfig()
        assert isinstance(cfg.enable_ucx, bool)
        assert isinstance(cfg.enable_shm, bool)
        assert isinstance(cfg.numa_aware, bool)
        assert isinstance(cfg.batch_size, int)
        assert isinstance(cfg.shm_name, str)
        assert isinstance(cfg.shm_size, int)

    def test_default_ucx_disabled(self):
        cfg = HPCConfig()
        assert cfg.enable_ucx is False

    def test_default_shm_enabled(self):
        cfg = HPCConfig()
        assert cfg.enable_shm is True

    def test_default_numa_disabled(self):
        cfg = HPCConfig()
        assert cfg.numa_aware is False

    def test_default_batch_size(self):
        cfg = HPCConfig()
        assert cfg.batch_size == 100

    def test_default_shm_name(self):
        cfg = HPCConfig()
        assert cfg.shm_name == "agi_dht_shm"

    def test_default_shm_size(self):
        cfg = HPCConfig()
        assert cfg.shm_size == 64 * 1024 * 1024

    @patch.dict(os.environ, {"AGI_DHT_UCX_ENABLED": "true"})
    def test_ucx_env_override(self):
        cfg = HPCConfig()
        assert cfg.enable_ucx is True

    @patch.dict(os.environ, {"AGI_DHT_BATCH_SIZE": "200"})
    def test_batch_size_env_override(self):
        cfg = HPCConfig()
        assert cfg.batch_size == 200
