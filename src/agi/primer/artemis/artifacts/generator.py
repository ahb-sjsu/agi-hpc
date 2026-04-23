# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Render Markdown handouts to PDF via pandoc.

Source handouts live alongside this module so they travel with the
code and can be diffed sensibly. Output PDFs are not committed — they
are regenerated per deploy to whichever artifact server ends up
fronting them (static file server, aiohttp endpoint, Caddy passthrough).

The renderer is intentionally thin: pandoc does the hard work. We
wrap it so tests can inject a fake subprocess and so callers get
structured errors instead of raw CalledProcessError.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("artemis.artifacts.generator")

HANDOUT_SOURCE_DIR = Path(__file__).resolve().parent / "handouts" / "source"


@dataclass(frozen=True)
class HandoutMeta:
    """Metadata parsed from the top of a handout .md file.

    Source files begin with a YAML-ish front-matter block::

        ---
        slug: session0_briefing
        title: Session Zero Briefing — MKS Halyard
        audience: all
        secrets: none
        ---

    ``slug`` is used as the output filename (``<slug>.pdf``). ``audience``
    is either ``all`` (every player gets it) or a specific ``player:<id>``
    (per-character handouts, S1-pre). ``secrets`` is either ``none`` or
    a comma-separated marker list used by the Keeper portal to filter
    what's safe to share.
    """

    slug: str
    title: str
    audience: str = "all"
    secrets: str = "none"
    path: Path | None = None


_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class HandoutError(RuntimeError):
    """Raised when a handout can't be parsed or rendered."""


def discover_handouts(source_dir: Path | None = None) -> list[HandoutMeta]:
    """Return every ``.md`` handout in ``source_dir`` with its metadata."""
    src = source_dir or HANDOUT_SOURCE_DIR
    if not src.is_dir():
        return []
    out: list[HandoutMeta] = []
    for path in sorted(src.glob("*.md")):
        if path.name.startswith("_") or path.name == "README.md":
            continue
        try:
            out.append(_parse_front_matter(path))
        except HandoutError as e:
            log.warning("skipping %s: %s", path.name, e)
    return out


def render_handout(
    meta: HandoutMeta,
    out_dir: Path,
    *,
    pandoc: str = "pandoc",
    runner: Callable[[list[str]], "subprocess.CompletedProcess"] | None = None,
) -> Path:
    """Render ``meta`` to ``out_dir/<slug>.pdf`` using pandoc.

    Raises :class:`HandoutError` on tool-missing or render failure.

    ``runner`` is dependency-injected so unit tests can assert the
    pandoc invocation without spawning a real process.
    """
    if meta.path is None or not meta.path.is_file():
        raise HandoutError(f"handout source missing: {meta}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{meta.slug}.pdf"

    if runner is None:
        if not shutil.which(pandoc):
            raise HandoutError(
                f"{pandoc!r} not found on PATH — install it "
                "(apt install pandoc texlive-xetex) or pass a runner"
            )
        runner = _default_runner

    # Keep the pandoc invocation font-agnostic — xelatex's default
    # (Latin Modern) is always present on texlive. Forcing Helvetica
    # etc. breaks on boxes that don't have the commercial font metrics
    # installed. Consumers that want custom typography should pass
    # their own template via front-matter later.
    cmd = [
        pandoc,
        str(meta.path),
        "-o",
        str(pdf_path),
        "--standalone",
        "--pdf-engine=xelatex",
        "--variable=geometry:margin=1in",
        "--metadata",
        f"title={meta.title}",
    ]
    try:
        result = runner(cmd)
    except FileNotFoundError as e:
        raise HandoutError(f"pandoc not runnable: {e}") from e
    if getattr(result, "returncode", 0) != 0:
        stderr = getattr(result, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise HandoutError(f"pandoc failed rc={result.returncode}: {str(stderr)[:300]}")
    if not pdf_path.is_file():
        raise HandoutError(f"pandoc reported success but {pdf_path} is missing")
    return pdf_path


# ─────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────


def _parse_front_matter(path: Path) -> HandoutMeta:
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        raise HandoutError("missing YAML front matter")
    block = match.group(1)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip().strip('"').strip("'")
    slug = fields.get("slug")
    title = fields.get("title")
    if not slug:
        raise HandoutError("front matter missing `slug`")
    if not title:
        raise HandoutError("front matter missing `title`")
    return HandoutMeta(
        slug=slug,
        title=title,
        audience=fields.get("audience", "all"),
        secrets=fields.get("secrets", "none"),
        path=path,
    )


def _default_runner(cmd: list[str]) -> "subprocess.CompletedProcess":
    return subprocess.run(cmd, capture_output=True, check=False)
