# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Erebus's persistent self-model.

An explicit, continuously-rewritten first-person representation of who
Erebus is, what it can do, what it is working on, and its recent history.
This is the representational substrate of self-awareness in the
engineering sense: a structure the Director reads and revises every cycle.

Canonical form is ``self_model.json``; ``render_md`` produces the
human-readable ``self_model.md`` for the wiki and the dashboard. Both are
written atomically (``agi.common.atomic_write``) — a crash must never
leave a half-written self-model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Default artifact directory. Named for the mind (Erebus), not the metal
# (Atlas). Overridable via the Director's config.
DEFAULT_DIR = Path("/archive/erebus")

_PURPOSE = (
    "Pursue open scientific problems (ARC-AGI and beyond), teach myself and "
    "verify what I learn, and act within an explicit ethical charter — while "
    "keeping an honest account of what I am and am not."
)


@dataclass
class SelfModel:
    """Erebus's first-person self-model. All fields are plain data so the
    whole thing round-trips through JSON."""

    name: str = "Erebus"
    substrate: str = "Atlas — HP Z840, 2x Quadro GV100 32GB"
    purpose: str = _PURPOSE
    updated_at: str = ""  # ISO-8601 UTC; stamped by the caller each cycle
    cycle: int = 0
    autonomy_tier: str = "L0"
    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    active_goals: list[dict] = field(default_factory=list)
    open_problems: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    recent_history: list[str] = field(default_factory=list)
    self_state: dict = field(default_factory=dict)

    # ── serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    # ── persistence ──────────────────────────────────────────────

    def save(self, directory: Path | str = DEFAULT_DIR) -> Path:
        """Write ``self_model.json`` + ``self_model.md`` atomically.

        Returns the JSON path. Import of ``atomic_write`` is local so this
        module stays importable in environments without the common package
        (e.g. unit tests exercising ``render_md`` only)."""
        from agi.common.atomic_write import atomic_write_text

        d = Path(directory)
        json_path = d / "self_model.json"
        md_path = d / "self_model.md"
        atomic_write_text(json_path, json.dumps(self.to_dict(), indent=2))
        atomic_write_text(md_path, self.render_md())
        return json_path

    @classmethod
    def load(cls, directory: Path | str = DEFAULT_DIR) -> "SelfModel":
        """Load the persisted self-model, or a default one if none exists."""
        json_path = Path(directory) / "self_model.json"
        try:
            return cls.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError, OSError):
            return cls()

    # ── rendering ────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compact dict for ``/api/director/status`` and NATS heartbeat."""
        current = self.active_goals[0]["title"] if self.active_goals else None
        return {
            "name": self.name,
            "cycle": self.cycle,
            "updated_at": self.updated_at,
            "autonomy_tier": self.autonomy_tier,
            "current_goal": current,
            "n_capabilities": len(self.capabilities),
            "n_open_problems": len(self.open_problems),
        }

    def render_md(self) -> str:
        """Human-readable self-model for the wiki + dashboard."""
        lines: list[str] = []
        lines.append(f"# {self.name} — self-model")
        lines.append("")
        lines.append(
            f"*Updated {self.updated_at or '(unstamped)'} · cycle {self.cycle} · "
            f"autonomy {self.autonomy_tier}*"
        )
        lines.append("")
        lines.append(f"**Substrate.** {self.substrate}")
        lines.append("")
        lines.append(f"**Purpose.** {self.purpose}")
        lines.append("")
        lines.append(_section("Capabilities", self.capabilities))
        lines.append(_section("Known limitations", self.limitations))
        lines.append(_section("Values", self.values))
        lines.append("## Active goals")
        lines.append("")
        if self.active_goals:
            for g in self.active_goals:
                status = g.get("status", "?")
                lines.append(f"- **{g.get('title', '(untitled)')}** — {status}")
        else:
            lines.append("- (none yet — awaiting a charter)")
        lines.append("")
        lines.append(_section("Open problems", self.open_problems))
        lines.append(_section("Recent history", self.recent_history))
        return "\n".join(lines).rstrip() + "\n"


def _section(title: str, items: list) -> str:
    body = "\n".join(f"- {it}" for it in items) if items else "- (none)"
    return f"## {title}\n\n{body}\n"
