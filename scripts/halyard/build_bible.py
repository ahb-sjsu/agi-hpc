#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Build the Halyard bible JSON from the wiki/ markdown tree.

Reads every ``.md`` under ``wiki/halyard/`` (or the ``--source``
path), parses the YAML frontmatter, and emits a JSON document
shaped for :class:`agi.primer.artemis.context.Bible.load` to
consume directly.

Each wiki file's frontmatter declares visibility per-AI:

    ---
    id:       setting/solar-system-2348
    title:    The Solar System in 2348
    artemis:  known | unknown | forbidden
    sigma4:   known | unknown | forbidden
    topic:    setting | faction | ship | crew | location | tech | mission | mythos
    tags:     [optional, hashtag-ish]
    ---

Output schema::

    {
      "schema_version": "1.0",
      "built_at":       "2026-04-24T22:15:00Z",
      "chunks": [
        {
          "id":    "setting/solar-system-2348",
          "tag":   "artemis_known",     # or sigma4_known, *_unknown, *_forbidden
          "title": "The Solar System in 2348",
          "text":  "Humanity lives from Luna to…",
          "topic": "setting",
          "tags":  ["overview", "politics"]
        },
        ...
      ],
      "forbidden_phrases": ["the Chamber", "Mi-go", …]
    }

Each wiki entry produces up to two chunks (one per AI's tag),
though if both AIs' visibility is ``known`` we emit under a
shared ``known`` tag pattern — the bible loader's filter handles
the rest.

Usage::

    python scripts/halyard/build_bible.py \
        [--source wiki/halyard] \
        [--out /archive/halyard/bible/halyard_bible.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    flags=re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)``. Missing FM → ({}, text)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_raw, body = m.group(1), m.group(2)
    # Tiny YAML-lite parser — we know the schema.
    fm: dict[str, Any] = {}
    for line in fm_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            # List.
            inner = v[1:-1]
            fm[k] = [s.strip() for s in inner.split(",") if s.strip()]
        else:
            fm[k] = v
    return fm, body.strip()


def _visibility(fm: dict[str, Any], which: str, default: str) -> str:
    v = fm.get(which, default)
    if v not in {"known", "unknown", "forbidden"}:
        raise ValueError(
            f"{fm.get('id', '?')}: invalid {which} visibility {v!r}"
        )
    return v


def _default_visibility(path: Path) -> str:
    """Directory-level default. mythos/ defaults to forbidden."""
    parts = {p.name for p in path.parents}
    if path.parent.name == "mythos":
        return "forbidden"
    return "known"


def _collect(source: Path) -> tuple[list[dict[str, Any]], list[str]]:
    chunks: list[dict[str, Any]] = []
    forbidden_phrases: list[str] = []

    md_files = sorted(p for p in source.rglob("*.md") if p.name != "README.md")
    for p in md_files:
        text = p.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        if not fm:
            print(f"warn: {p} has no frontmatter; skipped", file=sys.stderr)
            continue
        wiki_id = fm.get("id") or p.relative_to(source).with_suffix("").as_posix()
        title = fm.get("title") or wiki_id
        topic = fm.get("topic", p.parent.name)
        tags = fm.get("tags") or []
        default_v = _default_visibility(p)
        v_artemis = _visibility(fm, "artemis", default_v)
        v_sigma4 = _visibility(fm, "sigma4", default_v)

        # Body content is the retrievable text — title + content.
        body_full = f"# {title}\n\n{body}" if body else f"# {title}"

        # Emit one chunk per AI visibility. The Bible's tag scheme
        # uses <ai>_<known|unknown>; forbidden is a separate list.
        for which, v in (("artemis", v_artemis), ("sigma4", v_sigma4)):
            if v == "forbidden":
                # Lift forbidden-only content into the phrase ban list.
                forbidden_phrases.append(title)
                # A short acronym of the title also goes in (e.g.
                # "Starry Wisdom Church" → "Starry Wisdom").
                words = title.split()
                if len(words) >= 2:
                    forbidden_phrases.append(" ".join(words[:2]))
                continue

            chunks.append(
                {
                    "id": f"{wiki_id}@{which}",
                    "tag": f"{which}_{v}",
                    "title": title,
                    "text": body_full,
                    "topic": topic,
                    "tags": tags,
                }
            )

    # Deduplicate forbidden phrases.
    seen: set[str] = set()
    forbidden_phrases = [
        p for p in forbidden_phrases if not (p in seen or seen.add(p))
    ]
    return chunks, forbidden_phrases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="wiki/halyard",
        help="wiki tree to chunk (default: wiki/halyard).",
    )
    parser.add_argument(
        "--out",
        default="/archive/halyard/bible/halyard_bible.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out)

    if not source.is_dir():
        print(f"no such source dir: {source}", file=sys.stderr)
        return 1

    chunks, forbidden = _collect(source)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "built_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "chunks": chunks,
        "forbidden_phrases": forbidden,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    artemis_k = sum(1 for c in chunks if c["tag"] == "artemis_known")
    sigma4_k = sum(1 for c in chunks if c["tag"] == "sigma4_known")
    artemis_u = sum(1 for c in chunks if c["tag"] == "artemis_unknown")
    sigma4_u = sum(1 for c in chunks if c["tag"] == "sigma4_unknown")
    print(f"wrote {out}")
    print(f"  chunks total:      {len(chunks)}")
    print(f"  artemis_known:     {artemis_k}")
    print(f"  artemis_unknown:   {artemis_u}")
    print(f"  sigma4_known:      {sigma4_k}")
    print(f"  sigma4_unknown:    {sigma4_u}")
    print(f"  forbidden phrases: {len(forbidden)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
