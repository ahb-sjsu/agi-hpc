# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Render pregen characters (:mod:`.pregens`) as handout Markdown.

Each pregen becomes one ``.md`` file with the same front-matter
contract as every other handout — so they drop straight into the
:func:`.generator.discover_handouts` pipeline and render to PDF via
pandoc exactly like the Session 0 briefing.

Output naming: ``pregen_<id>.md`` (e.g. ``pregen_imogen.md``). The
HandoutMeta ``audience`` field is always ``all`` at this stage — the
pregens are public "pick-a-character" sheets, not player-assigned
secrets. (A per-player secrets companion is Keeper-audience and lives
in its own module.)
"""

from __future__ import annotations

from pathlib import Path

from .pregens import ROSTER, Pregen, Skill


def _slug_for(pregen: Pregen) -> str:
    return f"pregen_{pregen.id}"


def _title_for(pregen: Pregen) -> str:
    return f"Pregen — {pregen.name} ({pregen.role})"


def _skills_table(skills: tuple[Skill, ...]) -> str:
    rows = ["| Skill | Full | Half | Fifth |", "|---|---:|---:|---:|"]
    for s in skills:
        rows.append(f"| {s.name} | {s.value} | {s.half} | {s.fifth} |")
    return "\n".join(rows)


def _characteristics_table(p: Pregen) -> str:
    # Two-column layout — characteristics on the left, derived on the right.
    return (
        "| Characteristic | Value | Half | Fifth |"
        " | Derived | Value |\n"
        "|---|---:|---:|---:|"
        "|---|---:|\n"
        f"| STR | {p.str_} | {p.str_ // 2} | {p.str_ // 5} |"
        f" | Hit Points | {p.hp_max} |\n"
        f"| CON | {p.con} | {p.con // 2} | {p.con // 5} |"
        f" | Magic Points | {p.mp_max} |\n"
        f"| SIZ | {p.siz} | {p.siz // 2} | {p.siz // 5} |"
        f" | Sanity (start / max) | {p.san_start} / {p.san_max} |\n"
        f"| DEX | {p.dex} | {p.dex // 2} | {p.dex // 5} |"
        f" | Luck | {p.luck} |\n"
        f"| APP | {p.app} | {p.app // 2} | {p.app // 5} |"
        f" | Damage Bonus | {p.db} |\n"
        f"| INT | {p.int_} | {p.int_ // 2} | {p.int_ // 5} |"
        f" | Build | {p.build} |\n"
        f"| POW | {p.pow_} | {p.pow_ // 2} | {p.pow_ // 5} |"
        f" | Move Rate | {p.move} |\n"
        f"| EDU | {p.edu} | {p.edu // 2} | {p.edu // 5} |"
        f" | Age | {p.age} |\n"
    )


def render_pregen_markdown(pregen: Pregen) -> str:
    """Render one :class:`Pregen` to a full handout Markdown document.

    Deterministic + pure — no I/O. The caller decides whether to
    write it to disk (see :func:`write_pregen_handouts`) or pipe the
    string elsewhere.
    """
    front = (
        "---\n"
        f"slug: {_slug_for(pregen)}\n"
        f'title: "{_title_for(pregen)}"\n'
        "audience: all\n"
        "secrets: none\n"
        "---\n\n"
    )

    gear_lines = "\n".join(f"- {item}" for item in pregen.gear)

    body = (
        f"# {pregen.name}\n\n"
        f"**{pregen.role} · {pregen.occupation}**  \n"
        f"Age {pregen.age} · Origin {pregen.origin}\n\n"
        "## Characteristics & derived stats\n\n"
        f"{_characteristics_table(pregen)}\n"
        "## Skills\n\n"
        "_Core occupation + interest skills. All other skills start at "
        "the CoC 7e base._\n\n"
        f"{_skills_table(pregen.skills)}\n\n"
        "## Equipment & loadout\n\n"
        f"{gear_lines}\n\n"
        "## Background\n\n"
        f"{pregen.background}\n\n"
        "## Hook\n\n"
        f"{pregen.hook}\n\n"
        "## Appearance note\n\n"
        f"_{pregen.portrait_note}_\n"
    )
    return front + body


def write_pregen_handouts(
    out_dir: Path,
    *,
    roster: tuple[Pregen, ...] = ROSTER,
) -> list[Path]:
    """Write one Markdown file per pregen into ``out_dir``.

    Returns the list of file paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in roster:
        path = out_dir / f"{_slug_for(p)}.md"
        path.write_text(render_pregen_markdown(p), encoding="utf-8")
        written.append(path)
    return written


def roster_to_csv(roster: tuple[Pregen, ...] = ROSTER) -> str:
    """Dump the roster as a published-to-web CSV matching the sheet-poller schema.

    Used to seed the Google Sheet with the canonical pregens so the
    HUD + handouts stay in sync. The CSV columns mirror what the
    :mod:`agi.primer.artemis.sheets.parser` expects.
    """
    cols = [
        "id",
        "name",
        "role",
        "occupation",
        "age",
        "str",
        "con",
        "siz",
        "dex",
        "app",
        "int",
        "pow",
        "edu",
        "san",
        "san_max",
        "hp",
        "hp_max",
        "luck",
        "luck_max",
        "mp",
        "mp_max",
        "status",
        "notes",
    ]
    lines = [",".join(cols)]
    for p in roster:
        row = [
            p.id,
            p.name,
            p.role,
            p.occupation,
            str(p.age),
            p.str_,
            p.con,
            p.siz,
            p.dex,
            p.app,
            p.int_,
            p.pow_,
            p.edu,
            p.san,
            p.san_max,
            p.hp,
            p.hp_max,
            p.luck,
            p.luck_max,
            p.mp,
            p.mp_max,
            p.starting_status,
            "",
        ]
        lines.append(",".join(_csv_escape(str(cell)) for cell in row))
    return "\n".join(lines) + "\n"


def _csv_escape(cell: str) -> str:
    if any(c in cell for c in (",", '"', "\n")):
        return '"' + cell.replace('"', '""') + '"'
    return cell
