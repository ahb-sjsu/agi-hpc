#!/usr/bin/env python3
r"""
generate_protos.py

Generates Python gRPC stubs from all .proto files under ./proto
according to the AGI-HPC architecture in Appendix A.

Output will be generated into:
    src/agi/proto_gen/

This script:
 - Ensures protoc is available (system or grpc_tools)
 - Validates proto syntax
 - Generates *_pb2.py and *_pb2_grpc.py
 - Ensures __init__.py files exist
 - Provides --overwrite, --dry-run, --clean modes
 - Logs everything it does

Usage:
    python generate_protos.py
    python generate_protos.py --overwrite
    python generate_protos.py --clean
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

# ---------------------------------------------------------------------
# Utility logging
# ---------------------------------------------------------------------

def log(msg):
    print(f"[protos] {msg}")

def fatal(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------
# Detect protoc
# ---------------------------------------------------------------------

def find_protoc():
    """
    Try finding:
     1. system protoc
     2. python -m grpc_tools.protoc

    Returns (cmd_list, is_python_protoc)
    """
    # First try system protoc
    try:
        subprocess.run(["protoc", "--version"], check=True, capture_output=True)
        return ["protoc"], False
    except Exception:
        pass

    # Then try python grpc_tools
    try:
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "--version"],
                       check=True, capture_output=True)
        return [sys.executable, "-m", "grpc_tools.protoc"], True
    except Exception:
        pass

    fatal("Could not find protoc or grpc_tools.protoc. Install via pip install grpcio-tools.")


# ---------------------------------------------------------------------
# Clean generated directory
# ---------------------------------------------------------------------

def clean_out_dir(out_dir: Path):
    if out_dir.exists():
        log(f"Cleaning output dir: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Generate stubs
# ---------------------------------------------------------------------

def generate_stubs(proto_dir: Path, out_dir: Path, overwrite=False, dry_run=False):
    protoc_cmd, using_python = find_protoc()

    log(f"Using protoc command: {' '.join(protoc_cmd)}")

    proto_files = sorted(proto_dir.glob("*.proto"))
    if not proto_files:
        fatal(f"No .proto files found in {proto_dir}")

    log(f"Found {len(proto_files)} proto files:")
    for pf in proto_files:
        log(f"    - {pf}")

    # Ensure python package output structure
    (out_dir / "agi").mkdir(parents=True, exist_ok=True)
    (out_dir / "agi" / "proto_gen").mkdir(parents=True, exist_ok=True)
    (out_dir / "agi" / "__init__.py").touch()
    (out_dir / "agi" / "proto_gen" / "__init__.py").touch()

    # Generate stubs for each proto file
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

        # Execute protoc
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            fatal(f"protoc failed on {pf.name}: {e}")

        log(f"✓ Generated stubs for {pf.name}")

    log("All proto files compiled.")


# ---------------------------------------------------------------------
# Main CLI entrypoint
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting the output directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show commands but don't run protoc")
    ap.add_argument("--clean", action="store_true",
                    help="Delete output directory before generating")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    proto_dir = repo_root / "proto"
    out_dir = repo_root / "src"

    if not proto_dir.exists():
        fatal(f"Proto directory not found: {proto_dir}")

    # Output will be: src/agi/proto_gen/*
    gen_root = out_dir / "agi" / "proto_gen"

    if args.clean:
        clean_out_dir(gen_root)
    else:
        if gen_root.exists() and not args.overwrite and not args.dry_run:
            fatal(f"Output dir exists: {gen_root}\nUse --overwrite or --clean.")

    generate_stubs(proto_dir, out_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    log("Proto generation complete.")
    log(f"Generated files are under: {gen_root}")

if __name__ == "__main__":
    main()
