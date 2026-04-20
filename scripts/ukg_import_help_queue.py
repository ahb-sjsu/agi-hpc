#!/usr/bin/env python3
"""One-shot: import Erebus's help queue into the Unified Knowledge Graph as gaps.

Useful for bootstrapping the graph on an existing Atlas where the Primer
hasn't run yet, or for quickly syncing after a manual help-queue edit.
The Primer calls the same function (``import_help_queue``) at the top of
every tick — you normally don't need to run this by hand.

Usage:
    python3 scripts/ukg_import_help_queue.py
    python3 scripts/ukg_import_help_queue.py --help-queue /tmp/q.json --dry-run
    python3 scripts/ukg_import_help_queue.py --graph /tmp/g.jsonl --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agi.knowledge.gap_import import import_help_queue  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--help-queue",
        type=Path,
        default=None,
        help="Path to erebus_help_queue.json (default: /archive/neurogolf/erebus_help_queue.json)",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Graph JSONL path (default: KNOWLEDGE_GRAPH_PATH env or /archive/neurogolf/knowledge_graph.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be imported without touching the graph",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-entry log lines",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = import_help_queue(
        args.help_queue,
        graph_path=args.graph,
        dry_run=args.dry_run,
    )
    print(report.summary())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
