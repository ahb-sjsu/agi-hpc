# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Atomic file writes for crash-safe state persistence.

Stateful Atlas services (Primer, Erebus, help queue) write JSON state
files continuously. A plain ``path.write_text(...)`` is two syscalls
(truncate + write) and a crash between them leaves the file empty or
half-written. Readers ``try/except Exception: pass`` the parse error,
which means state is silently lost.

``atomic_write_text`` writes to a sibling tempfile, fsyncs, and then
atomically renames into place — the destination is either the old
contents or the fully-written new contents, never partial.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path | str, text: str, *, fsync: bool = True) -> None:
    """Write ``text`` to ``path`` atomically.

    Parameters
    ----------
    path:
        Destination file.  Parent directory is created if missing.
    text:
        Content to write.
    fsync:
        If True (default), fsync the tempfile before rename so the
        data survives a power loss, not just a process crash. Set
        False for hot-path writes where durability-vs-throughput
        tradeoff favors speed and the state is reconstructible.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent, prefix=p.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
