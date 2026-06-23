"""Render ARTEMIS handout Markdown files into PDF via pandoc.

Usage:
    python scripts/generate_handouts.py --out dist/
    python scripts/generate_handouts.py --slug session0_briefing
    python scripts/generate_handouts.py --list        # just list what would run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agi.primer.artemis.artifacts import (  # noqa: E402
    HANDOUT_SOURCE_DIR,
    HandoutError,
    discover_handouts,
    render_handout,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path("dist/handouts").resolve()),
        help="output directory for rendered PDFs (default: dist/handouts)",
    )
    ap.add_argument(
        "--slug",
        help="render a single handout by its front-matter slug",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list discovered handouts without rendering",
    )
    ap.add_argument(
        "--source",
        default=str(HANDOUT_SOURCE_DIR),
        help=f"handout source directory (default: {HANDOUT_SOURCE_DIR})",
    )
    args = ap.parse_args()

    source_dir = Path(args.source)
    handouts = discover_handouts(source_dir)

    if args.list:
        for h in handouts:
            print(f"  {h.slug:<28} {h.audience:<20} {h.title}")
        print(f"{len(handouts)} handout(s) in {source_dir}")
        return 0

    if args.slug:
        handouts = [h for h in handouts if h.slug == args.slug]
        if not handouts:
            print(f"no handout with slug={args.slug!r}", file=sys.stderr)
            return 1

    out_dir = Path(args.out)
    ok = 0
    for h in handouts:
        try:
            pdf = render_handout(h, out_dir)
            print(f"rendered  {h.slug:<28} -> {pdf}")
            ok += 1
        except HandoutError as e:
            print(f"failed    {h.slug:<28} -- {e}", file=sys.stderr)

    print(f"{ok}/{len(handouts)} handout(s) rendered into {out_dir}")
    return 0 if ok == len(handouts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
