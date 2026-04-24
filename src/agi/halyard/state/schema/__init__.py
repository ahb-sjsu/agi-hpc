# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Character-sheet schema + field-level access control.

Exports:

- :func:`load_schema` — returns the Draft 2020-12 JSON Schema dict.
- :func:`validate_sheet` — raise :class:`jsonschema.ValidationError`
  on invalid sheets.
- :mod:`.access` — maps JSON-paths to ``{public, player, keeper}``
  access tiers for patch-time authorization.
- :mod:`.patch` — minimal RFC 6902 JSON-Patch applier + authz gate.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = Path(__file__).parent / "character_sheet.schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Load and cache the character-sheet JSON Schema."""
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_sheet(sheet: dict[str, Any]) -> None:
    """Validate a sheet against the schema.

    Raises :class:`jsonschema.ValidationError` on failure. Returns
    ``None`` on success.
    """
    jsonschema.validate(sheet, load_schema())


__all__ = ["load_schema", "validate_sheet"]
