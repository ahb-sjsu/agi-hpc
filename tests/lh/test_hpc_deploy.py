# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.lh.hpc_deploy module."""

import subprocess

import pytest
from unittest.mock import MagicMock, patch, call

from agi.lh.hpc_deploy import HPCDeployConfig, SlurmLauncher, ApptainerRunner

# ---------------------------------------------------------------------------
# TestHPCDeployConfig
# ---------------------------------------------------------------------------


class TestHPCDeployConfig:
    """Tests for the HPCDeployConfig dataclass."""

    def test_default_values(self):
        """HPCDeployConfig defaults should match expected HPC settings."""
        cfg = HPCDeployConfig()
        assert cfg.scheduler == "slurm"
        assert cfg.partition == "gpu"
        assert cfg.nodes == 1
        assert cfg.gpus_per_node == 1
        assert cfg.cpus_per_task == 4
        assert cfg.memory_gb == 32
        assert cfg.time_limit == "01:00:00"
        assert cfg.container_runtime == "apptainer"
        assert cfg.container_image == ""

    def test_custom_values(self):
        """HPCDeployConfig should accept overridden fields."""
        cfg = HPCDeployConfig(
            scheduler="pbs",
            partition="cpu",
            nodes=8,
            gpus_per_node=4,
            cpus_per_task=16,
            memory_gb=128,
            time_limit="04:00:00",
        )
        assert cfg.scheduler == "pbs"
        assert cfg.partition == "cpu"
        assert cfg.nodes == 8
        assert cfg.gpus_per_node == 4
        assert cfg.cpus_per_task == 16
        assert cfg.memory_gb == 128
        assert cfg.time_limit == "04:00:00"

    def test_container_fields(self):
        """HPCDeployConfig should store container runtime and image."""
        cfg = HPCDeployConfig(
            container_runtime="singularity",
            container_image="/images/agi.sif",
        )
        assert cfg.container_runtime == "singularity"
        assert cfg.container_image == "/images/agi.sif"

    def test_directory_defaults(self):
        """HPCDeployConfig should have work_dir and log_dir defaults."""
        cfg = HPCDeployConfig()
        assert isinstance(cfg.work_dir, str)
        assert isinstance(cfg.log_dir, str)

    def test_modules_default(self):
        """HPCDeployConfig should default to cuda and python modules."""
        cfg = HPCDeployConfig()
        assert isinstance(cfg.modules, list)
        assert len(cfg.modules) == 2
        assert "cuda/12.0" in cfg.modules
        assert "python/3.12" in cfg.modules

    def test_modules_custom(self):
        """HPCDeployConfig should accept custom module lists."""
        cfg = HPCDeployConfig(modules=["openmpi/4.1"])
        assert cfg.modules == ["openmpi/4.1"]


# ---------------------------------------------------------------------------
# TestSlurmLauncher
# ---------------------------------------------------------------------------


class TestSlurmLauncher:
    """Tests for the SlurmLauncher class."""

    def test_init_default_config(self):
        """SlurmLauncher should create a default config when none is given."""
        launcher = SlurmLauncher()
        assert launcher._config is not None
        assert launcher._config.partition == "gpu"

    def test_init_custom_config(self):
        """SlurmLauncher should accept a custom HPCDeployConfig."""
        cfg = HPCDeployConfig(partition="debug", nodes=2, gpus_per_node=4)
        launcher = SlurmLauncher(config=cfg)
        assert launcher._config.partition == "debug"
        assert launcher._config.nodes == 2
        assert launcher._config.gpus_per_node == 4

    def test_generate_script_basic(self):
        """generate_script should produce a valid SLURM batch script."""
        cfg = HPCDeployConfig(
            partition="gpu",
            nodes=1,
            gpus_per_node=1,
            cpus_per_task=4,
            memory_gb=32,
            time_limit="01:00:00",
            log_dir="/scratch/logs",
            work_dir="/scratch",
            modules=["cuda/12.0"],
        )
        launcher = SlurmLauncher(config=cfg)
        script = launcher.generate_script("test_job", "python train.py")

        assert script.startswith("#!/bin/bash")
        assert "#SBATCH --job-name=test_job" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --gpus-per-node=1" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --mem=32G" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "module load cuda/12.0" in script
        assert "python train.py" in script

    def test_generate_script_with_container(self):
        """generate_script should wrap command in container exec when image is set."""
        cfg = HPCDeployConfig(
            container_image="/images/agi.sif",
            container_runtime="apptainer",
            gpus_per_node=2,
        )
        launcher = SlurmLauncher(config=cfg)
        script = launcher.generate_script("container_job", "python main.py")

        assert "apptainer exec" in script
        assert "--nv" in script
        assert "/images/agi.sif" in script
        assert "python main.py" in script

    def test_generate_script_overrides(self):
        """generate_script should allow kwarg overrides of config values."""
        launcher = SlurmLauncher()
        script = launcher.generate_script(
            "override_job",
            "echo hello",
            partition="debug",
            nodes=4,
            time_limit="00:30:00",
        )
        assert "#SBATCH --partition=debug" in script
        assert "#SBATCH --nodes=4" in script
        assert "#SBATCH --time=00:30:00" in script

    @patch("agi.lh.hpc_deploy.subprocess.run")
    @patch("agi.lh.hpc_deploy.tempfile.NamedTemporaryFile")
    def test_submit_returns_job_id(self, mock_tmpfile, mock_run):
        """submit should write a temp file, call sbatch, and return the job ID."""
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_script.sh"
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmpfile.return_value = mock_file

        mock_run.return_value = subprocess.CompletedProcess(
            args=["sbatch", "/tmp/test_script.sh"],
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
        )

        launcher = SlurmLauncher()
        job_id = launcher.submit("#!/bin/bash\necho hello")

        assert job_id == "12345"
        mock_run.assert_called_once()

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_cancel_success(self, mock_run):
        """cancel should return True when scancel succeeds."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["scancel", "12345"],
            returncode=0,
            stdout="",
            stderr="",
        )
        launcher = SlurmLauncher()
        assert launcher.cancel("12345") is True

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_cancel_failure(self, mock_run):
        """cancel should return False when scancel fails."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["scancel", "99999"],
            returncode=1,
            stdout="",
            stderr="Invalid job id",
        )
        launcher = SlurmLauncher()
        assert launcher.cancel("99999") is False

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_status_parses_sacct_output(self, mock_run):
        """status should parse sacct output into a structured dict."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["sacct"],
            returncode=0,
            stdout="12345|COMPLETED|0:0|00:05:30|node01\n",
            stderr="",
        )
        launcher = SlurmLauncher()
        info = launcher.status("12345")

        assert isinstance(info, dict)
        assert info["job_id"] == "12345"
        assert info["state"] == "COMPLETED"
        assert info["exit_code"] == "0:0"
        assert info["elapsed"] == "00:05:30"
        assert info["node_list"] == "node01"

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_status_unknown_when_sacct_fails(self, mock_run):
        """status should return UNKNOWN state when sacct fails."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["sacct"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        launcher = SlurmLauncher()
        info = launcher.status("99999")

        assert info["state"] == "UNKNOWN"
        assert info["job_id"] == "99999"


# ---------------------------------------------------------------------------
# TestApptainerRunner
# ---------------------------------------------------------------------------


class TestApptainerRunner:
    """Tests for the ApptainerRunner class."""

    def test_init_default_config(self):
        """ApptainerRunner should create a default config when none is given."""
        runner = ApptainerRunner()
        assert runner._config is not None
        assert runner._runtime == "apptainer"

    def test_init_custom_config(self):
        """ApptainerRunner should use the runtime from the config."""
        cfg = HPCDeployConfig(container_runtime="singularity")
        runner = ApptainerRunner(config=cfg)
        assert runner._runtime == "singularity"

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_build_success(self, mock_run):
        """build should return True on successful container build."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["apptainer", "build"],
            returncode=0,
            stdout="",
            stderr="",
        )
        runner = ApptainerRunner()
        result = runner.build("agi.def", "/images/agi.sif")

        assert result is True
        mock_run.assert_called_once_with(
            ["apptainer", "build", "/images/agi.sif", "agi.def"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_build_failure(self, mock_run):
        """build should return False when the build command fails."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["apptainer", "build"],
            returncode=1,
            stdout="",
            stderr="Build failed",
        )
        runner = ApptainerRunner()
        assert runner.build("bad.def", "/images/bad.sif") is False

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_run_with_gpu_and_binds(self, mock_run):
        """run should construct a command with --nv and --bind flags."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        runner = ApptainerRunner()
        result = runner.run(
            image="/images/agi.sif",
            command="python train.py",
            binds=["/data:/data", "/models:/models"],
            gpu=True,
        )

        assert result.returncode == 0
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "apptainer"
        assert cmd_args[1] == "run"
        assert "--nv" in cmd_args
        assert "--bind" in cmd_args
        assert "/data:/data" in cmd_args
        assert "/images/agi.sif" in cmd_args

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_run_without_gpu(self, mock_run):
        """run should omit --nv when gpu=False."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        runner = ApptainerRunner()
        runner.run(image="/images/agi.sif", command="echo hi", gpu=False)

        cmd_args = mock_run.call_args[0][0]
        assert "--nv" not in cmd_args

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_exec_calls_subprocess(self, mock_run):
        """exec should invoke the runtime exec command."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="hostname-01", stderr=""
        )
        runner = ApptainerRunner()
        result = runner.exec("/images/agi.sif", "hostname")

        assert isinstance(result, subprocess.CompletedProcess)
        assert result.stdout == "hostname-01"
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "apptainer"
        assert cmd_args[1] == "exec"
        assert "/images/agi.sif" in cmd_args

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_pull_success(self, mock_run):
        """pull should return True on successful image pull."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        runner = ApptainerRunner()
        result = runner.pull("docker://nvcr.io/nvidia/pytorch:latest", "/images/pt.sif")

        assert result is True
        mock_run.assert_called_once_with(
            [
                "apptainer",
                "pull",
                "/images/pt.sif",
                "docker://nvcr.io/nvidia/pytorch:latest",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("agi.lh.hpc_deploy.subprocess.run")
    def test_pull_failure(self, mock_run):
        """pull should return False when the pull command fails."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Pull failed"
        )
        runner = ApptainerRunner()
        assert runner.pull("docker://invalid", "/images/bad.sif") is False
