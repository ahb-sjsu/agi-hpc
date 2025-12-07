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
proto_fix.py

Idempotently patches all .proto files in ./proto to ensure:

1. syntax = "proto3";
2. correct package declaration:
      package agi.<filename>.v1;
3. syntax line appears first
4. package appears second
5. No duplicates are ever introduced

Run from repo root:
    python proto_fix.py
"""

from pathlib import Path
import re

PROTO_DIR = Path("proto")

def make_package_name(fname: str) -> str:
    """Create a package name based on filename."""
    base = fname.replace(".proto", "")
    # e.g., lh.proto → agi.lh.v1
    return f"package agi.{base}.v1;"

def fix_proto(path: Path):
    txt = path.read_text()
    lines = txt.splitlines()

    out = []
    changed = False

    # 1. Ensure syntax = "proto3";
    syntax_line = 'syntax = "proto3";'
    has_syntax = any(re.match(r'^\s*syntax\s*=\s*".+";', l) for l in lines)

    # 2. Ensure a correct package line
    package_line = make_package_name(path.name)
    has_package = any(l.strip().startswith("package ") for l in lines)

    # Build new header
    new_header = []
    if not has_syntax:
        new_header.append(syntax_line)
        changed = True
    if not has_package:
        new_header.append(package_line)
        changed = True

    # If header exists, ensure ordering: syntax first, then package
    # Remove old syntax/package and reinsert them in correct order
    filtered_lines = []
    for l in lines:
        if re.match(r'^\s*syntax\s*=\s*".+";', l):
            continue
        if l.strip().startswith("package "):
            continue
        filtered_lines.append(l)

    # Make sure header appears at the top
    out = new_header + filtered_lines

    # Only write if changes occurred
    if changed:
        print(f"[fix] {path}")
        path.write_text("\n".join(out) + "\n")
    else:
        print(f"[ok]  {path}")


def main():
    if not PROTO_DIR.exists():
        print("ERROR: No ./proto directory found.")
        return

    files = sorted(PROTO_DIR.glob("*.proto"))
    if not files:
        print("No .proto files found.")
        return

    print(f"Found {len(files)} proto files. Fixing...")
    for f in files:
        fix_proto(f)

    print("\nProto fix complete.")
    print("Run: python generate_protos.py --clean")


if __name__ == "__main__":
    main()
