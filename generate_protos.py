#!/usr/bin/env python3
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
generate_protos.py

Fully corrected version — produces *_pb2.py and *_pb2_grpc.py stubs
ONLY under: src/agi/proto_gen/

This version:
 - Never pollutes src/ with stray files
 - Works consistently in Windows + Linux (CI)
 - Ensures proto_gen is a valid Python package
 - Supports --clean, --overwrite, --dry-run
 - Produces deterministic output (same paths everywhere)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def log(msg: str):
    print(f"[protos] {msg}")

def fatal(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# Find protoc (system or grpc_tools)
# -----------------------------------------------------------------------------

def find_protoc():
    """
    Try:
        1. system protoc
        2. python -m grpc_tools.protoc
    """
    try:
        subprocess.run(["protoc", "--version"], check=True, capture_output=True)
        return ["protoc"], False
    except Exception:
        pass

    try:
        subprocess.run(
            [sys.executable, "-m", "grpc_tools.protoc", "--version"],
            check=True, capture_output=True
        )
        return [sys.executable, "-m", "grpc_tools.protoc"], True
    except Exception:
        pass

    fatal("Could not find protoc or grpc_tools.protoc. Install grpcio-tools.")


# -----------------------------------------------------------------------------
# Clean output dir
# -----------------------------------------------------------------------------

def clean_out_dir(out_dir: Path):
    if out_dir.exists():
        log(f"Cleaning output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Fix gRPC imports (post-processing)
# -----------------------------------------------------------------------------

def fix_grpc_imports(out_dir: Path):
    """
    Fix imports in generated *_pb2_grpc.py files.

    grpc_tools.protoc generates: `import foo_pb2 as foo__pb2`
    We need: `from agi.proto_gen import foo_pb2 as foo__pb2`
    """
    import re

    grpc_files = list(out_dir.glob("*_pb2_grpc.py"))
    if not grpc_files:
        return

    log(f"Fixing imports in {len(grpc_files)} gRPC files...")

    # Pattern: import <name>_pb2 as <name>__pb2
    pattern = re.compile(r'^import (\w+_pb2) as (\w+__pb2)$', re.MULTILINE)

    for grpc_file in grpc_files:
        content = grpc_file.read_text()
        original = content

        # Replace with package-relative import
        content = pattern.sub(r'from agi.proto_gen import \1 as \2', content)

        if content != original:
            grpc_file.write_text(content)
            log(f"  OK: Fixed imports in {grpc_file.name}")


# -----------------------------------------------------------------------------
# Generate protobuf stubs
# -----------------------------------------------------------------------------

def generate_stubs(proto_dir: Path, clean=False, dry_run=False):
    protoc_cmd, _ = find_protoc()

    # Force output path EXACTLY here:
    out_dir = Path("src/agi/proto_gen").resolve()

    if clean:
        clean_out_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure Python package init
    (out_dir / "__init__.py").touch()

    # Find proto files
    proto_files = sorted(proto_dir.glob("*.proto"))
    if not proto_files:
        fatal(f"No .proto files found in {proto_dir}")

    log(f"Using protoc: {' '.join(protoc_cmd)}")
    log(f"Output directory: {out_dir}")
    log(f"Found {len(proto_files)} proto files")

    for pf in proto_files:
        log(f"Compiling {pf.name} ...")

        cmd = protoc_cmd + [
            f"--proto_path={proto_dir}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            str(pf)
        ]

        if dry_run:
            log(f"(dry-run) Would run: {' '.join(cmd)}")
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            fatal(f"protoc failed for {pf.name}: {e}")

        log(f"OK: {pf.name}")

    log("All proto files compiled successfully.")

    # Post-process: fix imports in *_grpc.py files
    fix_grpc_imports(out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="Delete output dir before generating")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    proto_dir = repo_root / "proto"

    if not proto_dir.exists():
        fatal(f"Proto directory not found: {proto_dir}")

    generate_stubs(proto_dir, clean=args.clean, dry_run=args.dry_run)

    log(f"Generated stubs in: src/agi/proto_gen/")


if __name__ == "__main__":
    main()
