# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Canonical pregenerated characters for the Nithon expedition.

The five roles match the placeholders already wired into the player
table UI (``PLACEHOLDER_CHARS`` in ``deploy/web/table/table.js``).
Stats are sized for Call of Cthulhu 7e: characteristics in 40–80
range (standard distribution, occasional 80s for class skills),
skills tuned to each character's occupation niche.

This module is the single source of truth. Downstream consumers:

  * :func:`.pregen_handouts.render_pregen_markdown` renders a 2-page
    handout per pregen (Session 1 pre-game ship-out) via the shared
    :func:`.generator.render_handout` pipeline.
  * The S1b Google-Sheet seed CSV is generated from this roster so
    the live HUD + the handouts never diverge.
  * The Keeper-only companion (sealed-section secrets) lives in a
    separate, Keeper-audience data file — it is *not* in here.

No cosmic-horror spoilers live in these pregens. Every "hook" is
something the character knows, not something waiting for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    """A single skill line-item on the pregen sheet."""

    name: str
    value: int  # 0..95

    @property
    def half(self) -> int:
        return self.value // 2

    @property
    def fifth(self) -> int:
        return self.value // 5


@dataclass(frozen=True)
class Pregen:
    """A Call of Cthulhu 7e pregenerated character.

    Frozen so it can be shared across threads (used by the sheets
    poller, the handout generator, and the HUD seed at the same time).
    """

    id: str
    name: str
    role: str
    occupation: str
    age: int
    origin: str
    # Characteristics — CoC 7e scale
    str_: int
    con: int
    siz: int
    dex: int
    app: int
    int_: int
    pow_: int
    edu: int
    # Derived
    hp_max: int
    mp_max: int
    san_start: int  # equal to POW on a fresh character
    luck: int
    db: str  # e.g. "+1d4", "0", "-1"
    build: int
    move: int
    # Lists
    skills: tuple[Skill, ...]
    gear: tuple[str, ...]
    # Prose
    background: str
    hook: str
    portrait_note: str
    # Optional starting condition tweaks (e.g. mild sanity loss or
    # carryover injury from a prior mission). Default is fresh.
    starting_status: str = "ok"

    # ── convenience derived ─────────────────────────────────────

    @property
    def hp(self) -> int:
        """Current HP — pregens start undamaged."""
        return self.hp_max

    @property
    def mp(self) -> int:
        return self.mp_max

    @property
    def san(self) -> int:
        return self.san_start

    @property
    def san_max(self) -> int:
        # Per CoC 7e: SAN max = 99 minus Cthulhu Mythos. Fresh PCs = 99.
        return 99

    @property
    def luck_max(self) -> int:
        # House rule: 99. Most tables use this cap.
        return 99

    @property
    def mov(self) -> int:  # alias for the HUD's expected key
        return self.move

    # ── serialization ──────────────────────────────────────────

    def to_hud_dict(self) -> dict[str, object]:
        """Minimal dict shape the HUD sheet poller understands.

        Keys match ``agi.primer.artemis.sheets.parser.row_for_hud``'s
        contract so the pregen roster can be served as a drop-in CSV
        for dev / offline play.
        """
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "san": self.san,
            "san_max": self.san_max,
            "hp": self.hp,
            "hp_max": self.hp_max,
            "luck": self.luck,
            "luck_max": self.luck_max,
            "mp": self.mp,
            "mp_max": self.mp_max,
            "status": self.starting_status,
        }


# ─────────────────────────────────────────────────────────────────
# Roster — five canonical pregens, matching the HUD placeholder IDs.
# ─────────────────────────────────────────────────────────────────


def _skill(name: str, value: int) -> Skill:
    return Skill(name=name, value=value)


IMOGEN_ROTH = Pregen(
    id="imogen",
    name="IMOGEN ROTH",
    role="Expedition Lead",
    occupation="Licensed Expedition Commander (Mao-Kwikowski)",
    age=47,
    origin="Ceres (Belt)",
    str_=55,
    con=65,
    siz=60,
    dex=60,
    app=65,
    int_=75,
    pow_=70,
    edu=70,
    hp_max=12,
    mp_max=14,
    san_start=70,
    luck=55,
    db="0",
    build=0,
    move=7,
    skills=(
        _skill("Command", 70),
        _skill("Persuade", 65),
        _skill("Navigate", 60),
        _skill("Pilot (Spacecraft)", 60),
        _skill("Astrogation", 55),
        _skill("Credit Rating", 55),
        _skill("Listen", 60),
        _skill("Spot Hidden", 55),
        _skill("Psychology", 55),
        _skill("First Aid", 40),
        _skill("Firearms (Handgun)", 45),
        _skill("Belter Patois", 50),
    ),
    gear=(
        "Mao-Kwikowski command harness with sidearm holster",
        "Polarized handheld, Keeper-level comms access",
        "Sidearm (10 mm service pistol, 12 rounds)",
        "Personal kit: data-slate, stims, two-day EVA pack",
    ),
    background=(
        "Imogen has been pushing the outer-rim trade lanes since before the Ring "
        "Gate panic — first as a pilot under an OPA charter, then as a salvage "
        "lead, finally under Mao-Kwikowski's deep-survey division. She has the "
        "cantankerous pragmatism of a working Belter and the paperwork habits "
        "of someone who has had their license pulled twice and earned it back. "
        "The crew listens to her because she's earned it, not because the "
        "contract says they have to."
    ),
    hook=(
        "The last Mao-Kwikowski mission she led — a routine biological survey in "
        "2367 — returned short one crewmember. The incident report was closed. "
        "Imogen has read it every month since. She volunteered for the Nithon "
        "run because she does not believe in coincidences."
    ),
    portrait_note=(
        "Early-50s Belter features: tall, lean, close-cropped silvering hair, "
        "a faint Ceres tattoo at the temple. Uniform is clean, no insignia of "
        "rank — she doesn't need them."
    ),
)


ERIK_SULLIVAN = Pregen(
    id="sully",
    name="ERIK SULLIVAN",
    role="Chief Engineer",
    occupation="Engineer, Epstein drive + auxiliary systems",
    age=38,
    origin="Mars (MCRN veteran)",
    str_=70,
    con=75,
    siz=65,
    dex=55,
    app=50,
    int_=75,
    pow_=55,
    edu=75,
    hp_max=14,
    mp_max=11,
    san_start=55,
    luck=50,
    db="+1d4",
    build=1,
    move=7,
    skills=(
        _skill("Mechanical Repair", 80),
        _skill("Electrical Repair", 70),
        _skill("Electronics", 70),
        _skill("Engineering (Reactors)", 65),
        _skill("Operate Heavy Machinery", 60),
        _skill("Computer Use", 55),
        _skill("Pilot (Spacecraft)", 45),
        _skill("Listen", 55),
        _skill("Spot Hidden", 50),
        _skill("Firearms (Rifle)", 45),
        _skill("Swim", 40),
        _skill("Martian Standard", 55),
    ),
    gear=(
        "Engineer's multitool (magnetic, EVA-rated)",
        "Full pocket-tool loadout + replacement electronics packs",
        "Half-size sidearm (keeps clear of his work)",
        "Personal kit: caffeine inhaler, replacement suit seals",
    ),
    background=(
        "Sully came up through the Martian Congressional Republic Navy's engineering "
        "track — ten years keeping Epstein drives healthy on light cruisers before "
        "the attrition of the MCRN post-Ring collapse pushed him to civilian work. "
        "Mao-Kwikowski contracted him on the recommendation of the Halyard's "
        "previous chief engineer. He took the job quickly and didn't ask too many "
        "questions about what happened to his predecessor."
    ),
    hook=(
        "His predecessor on the Halyard vanished during the last Nithon survey. "
        "He left behind annotated schematics, a half-finished diagnostic tool, "
        "and a list of drive-telemetry oddities that Sully has been working "
        "through in his bunk. He hasn't told anyone yet."
    ),
    portrait_note=(
        "Late-30s Martian, stocky build from half-g childhood, ink-black hair "
        "kept buzzed short. Grease smudges are essentially part of his "
        "complexion. Hand-stitched kit modifications on his coveralls."
    ),
)


ASTA_NORDQUIST = Pregen(
    id="asta",
    name="ASTA NORDQUIST",
    role="Medical Officer",
    occupation="Ship's Surgeon + research xenobiologist",
    age=41,
    origin="Stockholm (Earth)",
    str_=50,
    con=55,
    siz=55,
    dex=60,
    app=65,
    int_=80,
    pow_=65,
    edu=80,
    hp_max=11,
    mp_max=13,
    san_start=65,
    luck=60,
    db="0",
    build=0,
    move=7,
    skills=(
        _skill("Medicine", 75),
        _skill("First Aid", 80),
        _skill("Psychology", 65),
        _skill("Psychoanalysis", 50),
        _skill("Science (Biology)", 75),
        _skill("Science (Pharmacy)", 60),
        _skill("Spot Hidden", 60),
        _skill("Library Use", 65),
        _skill("Computer Use", 55),
        _skill("Persuade", 50),
        _skill("Listen", 55),
        _skill("Language (Swedish)", 70),
    ),
    gear=(
        "Mao-Kwikowski medical loadout (autocauter, trauma kit, field surgery)",
        "Personal microscope + centrifuge for sample work",
        "Standard pharmacy rack (stims, analgesics, sedatives)",
        "Sidearm (kept in the med bay, rarely on her belt)",
    ),
    background=(
        "Asta left Earth trauma-surgery rotations for Mars-orbit research "
        "postings in her early thirties — a quieter schedule and more "
        "interesting cases. Her published work on radiation-aberrant "
        "microfauna in Jovian subsurface ice put her on Mao-Kwikowski's "
        "xenobiology shortlist. She prefers the Halyard to a lab because the "
        "Halyard goes places samples come from."
    ),
    hook=(
        "In 2366 she lost a patient under circumstances her report labeled "
        "'complications of acute radiation exposure.' What she did not write "
        "down: the patient's vitals stabilized for fourteen minutes after "
        "flatline. She logged the event in a private journal and told no one."
    ),
    portrait_note=(
        "Early-40s Scandinavian, calm-eyed, practical short hair, minimal "
        "jewelry. Her white coat is almost always over her shipboard fatigues "
        "even when she isn't in the med bay."
    ),
)


ARLO_VANCE = Pregen(
    id="arlo",
    name="ARLO VANCE",
    role="Surface Ops / EVA",
    occupation="Surface operations, heavy EVA, recovery",
    age=33,
    origin="Pallas (Belt)",
    str_=80,
    con=75,
    siz=70,
    dex=70,
    app=50,
    int_=60,
    pow_=50,
    edu=55,
    hp_max=15,
    mp_max=10,
    san_start=50,
    luck=45,
    db="+1d4",
    build=1,
    move=9,
    skills=(
        _skill("Climb", 70),
        _skill("Jump", 60),
        _skill("Spot Hidden", 70),
        _skill("Listen", 60),
        _skill("Dodge", 60),
        _skill("Firearms (Rifle)", 65),
        _skill("Firearms (Handgun)", 60),
        _skill("Mechanical Repair", 55),
        _skill("Operate Heavy Machinery", 55),
        _skill("Survival (Vacuum)", 60),
        _skill("First Aid", 45),
        _skill("Belter Patois", 60),
    ),
    gear=(
        "Hardshell deep-EVA suit (12-hour rated)",
        "Breaching kit: plasma cutter, mag-anchors, salvage tethers",
        "Rifle (11 mm anti-debris, 3 mags)",
        "Survival belt: emergency beacons, O₂ bottle, patch kit",
    ),
    background=(
        "Arlo grew up in Pallas shipyards, worked salvage crews out of the outer "
        "Belt, and has the scars to prove most of it. He took the Mao-Kwikowski "
        "contract because it pays better than salvage and because he wanted "
        "out of a particular circuit of creditors in the middle Belt. He is "
        "the crewmember most likely to be first through a breached airlock and "
        "the last one still willing to go outside when things get strange."
    ),
    hook=(
        "There's a salvage job on his record nobody in the Belt will discuss — "
        "a three-week contract in the trans-Neptunian zone that he finished "
        "alone. He keeps a charm welded to his suit. He won't say what it "
        "was, and nobody who was there is around to corroborate."
    ),
    portrait_note=(
        "Early-30s Belter, tall, wiry-muscled, dark eyes. Short dreadlocks, a "
        "network of faint scars across his jaw and forearms. Prefers a "
        "half-dressed jumpsuit indoors; the suit is always within arm's reach."
    ),
)


SAOIRSE_KELLEHER = Pregen(
    id="saoirse",
    name="SAOIRSE KELLEHER",
    role="Rad / Atmo Chemistry",
    occupation="Radiation + atmospheric chemist",
    age=29,
    origin="Dublin (Earth)",
    str_=45,
    con=55,
    siz=55,
    dex=65,
    app=60,
    int_=80,
    pow_=65,
    edu=75,
    hp_max=11,
    mp_max=13,
    san_start=65,
    luck=40,
    db="0",
    build=0,
    move=8,
    skills=(
        _skill("Science (Chemistry)", 75),
        _skill("Science (Physics)", 65),
        _skill("Science (Pharmacy)", 55),
        _skill("Electronics", 60),
        _skill("Computer Use", 65),
        _skill("Spot Hidden", 55),
        _skill("Listen", 55),
        _skill("Library Use", 70),
        _skill("First Aid", 45),
        _skill("Credit Rating", 35),
        _skill("Language (Irish Gaelic)", 55),
        _skill("Language (Latin)", 40),
    ),
    gear=(
        "Portable gas chromatograph + spectroscopy loadout",
        "Radiation survey kit (dosimetry, gamma + neutron counters)",
        "Sample-return cases (sealed, rad-shielded)",
        "Handheld with science-journal archives synced",
    ),
    background=(
        "Saoirse took a PhD track at Trinity College in cosmic-ray aerosol "
        "chemistry, two years in. Her thesis drifted from the institutional "
        "plan; her advisors called her speculative branches 'fringe' and her "
        "funding dried. Mao-Kwikowski hired her for field instrumentation "
        "work. Nobody has yet told her the Nithon signal overlaps the frequency "
        "bands her thesis flagged as anomalous."
    ),
    hook=(
        "She took this contract in part because Mao-Kwikowski was willing to "
        "ignore the reason Trinity suspended her final year — she has since "
        "declined to discuss it with anyone. She carries a paper notebook "
        "that no one on the ship has opened, including her."
    ),
    portrait_note=(
        "Late-20s Irish, dark red hair, freckled, wry-mouthed. Wears her "
        "science coveralls everywhere; swapped the Mao-Kwikowski badge for a "
        "plain grey patch on her first day."
    ),
    starting_status="shaken",  # the work already weighs on her
)


ROSTER: tuple[Pregen, ...] = (
    IMOGEN_ROTH,
    ERIK_SULLIVAN,
    ASTA_NORDQUIST,
    ARLO_VANCE,
    SAOIRSE_KELLEHER,
)


def by_id(pid: str) -> Pregen | None:
    """Lookup helper; returns ``None`` on miss."""
    for p in ROSTER:
        if p.id == pid:
            return p
    return None


def iter_roster() -> Iterable[Pregen]:
    """Iterator over the canonical roster. Useful for CSV seeding."""
    return iter(ROSTER)
