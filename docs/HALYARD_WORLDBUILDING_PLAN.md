# Halyard Worldbuilding & Illustration Plan

This is the multi-phase plan for the campaign's worldbuilding
expansion. Each phase has concrete deliverables, illustration
strategy, and an estimated turn budget. Phases run sequentially
unless otherwise noted; nothing in this plan ships for Session 0
tonight — it's the post-launch roadmap.

## Image generation — confirmed infrastructure

**NRP LLM gateway: NO image generation.** Probed
``/v1/images/generations``; returns "No matching route found."
NRP is LLM-only (gpt-oss, glm-4.7, kimi, qwen3, minimax-m2,
etc.). Confirmed.

**Atlas: full local image-generation capability.**
- ``diffusers 0.37.1`` already installed in ``/home/claude/env``
- ``torch 2.10.0+cu128`` with CUDA available
- Two Quadro GV100s, 32 GB VRAM free on each
- Hugging Face cache mounted; models can be pulled on demand
- SDXL or FLUX renders will run in 8–15 s/image

**Two illustration tracks, used in parallel:**

1. **SVG / programmatic** — schematics, exploded views, comparison
   charts, organizational diagrams, maps, weapon side-views,
   ship cross-sections, deck plans, faction org charts. Fast
   (sub-second per image), license-clean, edit-friendly,
   appropriate for technical content.
2. **SDXL on Atlas** — atmospheric concept art for locations,
   character portraits for NPCs, mood pieces for Mythos
   entries, scene illustrations for in-fiction documents.
   Slower (10–15 s/image) but richer; needed for tone-setting.

Both tracks emit PNGs into ``wiki/halyard/art/`` keyed by entry
id, referenced from markdown via plain ``![alt](art/...)``
links. The bible builder ignores image references in chunk
text; AI retrieval gets the prose, players see the prose +
image.

## Phase 1 — Mythos cosmology completion (this turn)

The Gate at Nithon and the Caretakers are load-bearing for
campaign Acts 5–6. Without them, the late-act narrative has no
mechanical anchor and the validator's forbidden-phrase list is
incomplete.

**Deliverables (this turn):**
- ``mythos/the-gate.md`` — the Nithon resonator, mechanism,
  history, activation protocol
- ``mythos/the-caretakers.md`` — what's at the other end of
  the gate, their interest in the Halyard expedition, the
  Act 5 Request scene
- ``mythos/shoggoths.md`` — Elder Thing servitors, their
  current presence in the campaign
- ``mythos/starry-wisdom-church.md`` — the public-front
  Mythos cult on Earth and Luna
- ``mythos/the-hollow-hand.md`` — anti-Chamber resistance
  with its own Mythos contamination

After this turn, the Mythos tier has 9 entries totaling ~50
pages and the validator's forbidden-phrase list covers every
significant Mythos vocabulary item.

**Illustrations:** Phase 1 ships text-only. Mythos entries get
SDXL atmospheric art in Phase 5 batch.

**Turn budget:** 1 turn (this one).

## Phase 2 — Regions & cities deep dives

Expand the major locations beyond their current overview-level
treatment.

**Earth** (~10 entries):
- Regions: Africa, Asia, Europe-Anatolia, Americas, Oceania,
  Antarctic Industrial Zone (6 entries, ~5 pp each)
- Cities: Shanghai, Mumbai, São Paulo, NYC arcology (4
  entries, ~4 pp each)

**Mars** (~10 entries):
- Regions: Hellas, Tharsis, Valles Marineris, Northern
  Lowlands, Southern Highlands, Polar Caps (6 entries)
- Cities: Olympus, Pavonis, Coprates, Acidalia (4 entries)

**Belt** (~10 entries):
- Bodies: Vesta, Pallas, Hygiea (3 entries)
- Jovian system: Ganymede, Callisto, Europa, Io (4 entries)
- The Trojan habitats (1 entry)
- The Kuiper-edge stations (1 entry)
- The spinner subculture (1 entry)

**Outer system** (~5 entries):
- Saturnian moons (Iapetus, Hyperion, Enceladus) (3 entries)
- Trans-Neptunian survey infrastructure (1 entry)
- The Neptune trailing point (1 entry)

**Luna** (~5 entries):
- Cities: Tycho, Aristarchus, Copernicus, Kepler-Reiner (4
  entries)
- The Lunar Industrial Belt as infrastructure (1 entry)

Total Phase 2: ~40 entries, ~180 pages.

**Illustrations:** Each region or city gets:
- 1 SVG location map (top-down, schematic, ~200 lines)
- 1 SVG district/zone diagram where relevant
- 1–2 SDXL atmospheric concept renders

**Turn budget:** 8–12 turns.

## Phase 3 — Remaining factions

Build out the faction roster. Existing: united-nations,
mao-kwikowski, opa, protogen.

**Open factions to build:**
- mcr (Martian Congressional Republic — domestic structure)
- unn (already drafted as Earth institution; reformat)
- mcrn (already drafted as Mars institution; reformat)
- lloyds-of-london (deep-range insurance underwriting)
- fsec (already drafted as Earth institution; reformat)
- martian-intelligence-branch
- opa-kuiper-division
- catholic-church-2348
- luna-temple-buddhism
- secular-humanist-fellowships
- ifa-revival
- council-of-the-confluences (Islamic)
- mars-atmospheric-authority
- saturnian-naval-coordination-authority
- universities-as-institutions (UN universities, Mars,
  Lagos)
- the-merchant-marines (Belt unions)
- the-spinners-confederation
- the-engineers-syndicate

Total Phase 3: ~18 entries, ~80 pages.

**Illustrations:** Each faction gets:
- 1 SVG organizational chart
- 1 SVG insignia / flag / uniform reference
- 1 SDXL identity piece where useful (a character archetype,
  a location, a ceremony)

**Turn budget:** 4–6 turns.

## Phase 4 — Tech & equipment guides

The biggest single deliverable. The campaign is hard-SF + cosmic
horror; PCs need to know what their tech does. Each guide is
4–10 pages, fully illustrated.

**Tier A — General reference:**
- ``tech/state-of-the-art-2348.md`` — overview, what's possible
  in 2348 vs. 2025, the trend lines
- ``tech/computing-and-ai.md`` — handhelds, ship systems,
  expert-system AIs (the kind ARTEMIS and SIGMA-4 are), LLM
  history in 2348, the regulatory framework
- ``tech/comms-and-networking.md`` — laser comms, light-speed
  delay management, encrypted relay systems, the Outer Authority
  beacon network

**Tier B — Combat & security:**
- ``tech/civilian-firearms.md`` — handguns, rifles, the legal
  framework, common models in 2348
- ``tech/military-firearms.md`` — UNN/MCRN small arms,
  ship-mounted PDCs, naval railguns, missile families
- ``tech/less-lethal.md`` — gas, stun, electric, sonic;
  shipboard-safe variants
- ``tech/grenades-and-throwables.md`` — frag, flash, gas,
  sonic, breach charges; vacuum-rated variants
- ``tech/civilian-armor.md`` — civilian body armor (rare);
  EVA suits as armor (universal)
- ``tech/military-armor.md`` — UNN power-armor, MCRN
  exoskeleton, Belt-derived light suits
- ``tech/spy-tech.md`` — surveillance, counter-surveillance,
  data ex-fil; the FSec / MCR-Intelligence toolkit
- ``tech/cyber-security.md`` — encryption standards, attack
  surfaces, the LiveKit / NATS / station-control vulnerabilities

**Tier C — Body & medicine:**
- ``tech/medical-tech.md`` — emergency medicine, surgery,
  sickbay equipment; the *Halyard*'s capabilities
- ``tech/regrowth-gel.md`` — what it is, how it works, what
  it can and cannot regrow, contraindications
- ``tech/implants-and-cybernetics.md`` — the regulatory
  framework, common implants (sensory, medical, communication),
  rare implants (neural-interface, military-grade)
- ``tech/biotech.md`` — gene therapy, synthetic organs,
  longevity treatments
- ``tech/nanotech.md`` — what's real in 2348 (mostly
  industrial), what's fiction (still fiction)
- ``tech/sanity-treatment.md`` — clinical SAN treatment in
  2348, the Lunar Temple meditative protocols, drug regimens,
  what does and doesn't work

**Tier D — Vehicles & mobility:**
- ``tech/vehicles-atmospheric.md`` — Earth/Mars surface and
  flight vehicles
- ``tech/vehicles-surface.md`` — rovers, surface vehicles for
  Mars/Luna/Belt-bodies
- ``tech/vehicles-space.md`` — civilian spacecraft, common
  freighters, courier ships, the Halyard's sister classes
- ``tech/eva-and-vacuum.md`` — EVA suits, vacuum survival,
  station emergency protocols

**Tier E — Specialty:**
- ``tech/drones.md`` — civilian, industrial, surveillance,
  military
- ``tech/cyberpunk-tech.md`` — the underground tech the
  legitimate guides won't cover; black-market modifications,
  illegal implants, off-record networks
- ``tech/encryption-deep.md`` — the cryptographic
  foundations, what's secure, what's quietly broken
- ``tech/mythos-tech.md`` — Keeper-only; brain-cylinders,
  Yithian-archive interfaces, Elder Thing artifacts, gate
  mechanisms

Total Phase 4: ~25 entries, ~150 pages.

**Illustrations:** This phase is illustration-heavy. Per entry:
- 2–4 SVG schematics (item drawings, exploded views, charts)
- 1 SVG comparison table where applicable
- 1 SDXL concept render for tonal entries (cyberpunk-tech,
  mythos-tech, sanity-treatment)

Each weapon entry gets a side-view SVG + spec-sheet table.
Each implant entry gets an anatomical diagram. Each vehicle
entry gets a 3-view SVG (top, side, front).

**Turn budget:** 10–15 turns.

## Phase 5 — Illustration infrastructure & batch

Once the prose foundation is laid (Phases 1–4), generate
illustrations in batch.

**Step 1: SDXL setup on Atlas (one-time).**
- Pull SDXL or FLUX-dev to ``/home/claude/.cache/huggingface/``
  (~7 GB, ~5 min)
- Write ``scripts/halyard/art_render.py`` that takes a YAML
  prompt list and emits PNGs
- Wire ``/halyard art <prompt>`` skill subcommand for ad-hoc
  generation
- ~30 min of work

**Step 2: SVG generators.**
- ``scripts/halyard/svg_weapon_schematic.py`` — generates
  side-view + spec-sheet SVGs from a weapon spec dict
- ``scripts/halyard/svg_org_chart.py`` — faction
  organizational charts
- ``scripts/halyard/svg_location_map.py`` — top-down map SVGs
- ``scripts/halyard/svg_comparison_chart.py`` — comparison
  tables (range, ROF, damage, etc.)
- Each generator: 1–2 hours of dev

**Step 3: Batch.**
- Walk all wiki entries that need illustrations
- For each, run the appropriate SVG generators + queue SDXL
  prompts
- Render overnight on Atlas (probably ~4–6 hours of GPU time
  for ~200 atmospheric pieces + a few hundred SVGs)
- Update wiki entries with image references

**Turn budget:** 3–4 turns (mostly setup; the batch runs
unattended).

## Phase 6 — Slide deck updates

Once the wiki has substantial Earth/Mars/Belt/Luna depth, fold
the player-facing material back into the Block C presentation
deck.

**Updates:**
- Add 6 new content slides: one per major region (Earth,
  Mars, Luna, Belt, Outer system) + one for the contract /
  consortium context
- Replace the current placeholder pull-quotes with material
  quoted from the wiki entries (so the slide deck and the
  wiki are in sync)
- Drop in SDXL atmospheric renders generated in Phase 5
- Update speaker notes with reference pointers into the wiki
  entries (so during play the Keeper can drill down on
  player questions)

The result is a presentation deck with ~14 content slides
backed by ~200 pages of wiki depth.

**Turn budget:** 1–2 turns.

## Total scope

| Phase | Deliverables | Pages | Turns |
|---|---|---|---|
| 1 — Mythos completion | 5 entries | ~50 | 1 |
| 2 — Regions & cities | ~40 entries | ~180 | 8–12 |
| 3 — Factions | ~18 entries | ~80 | 4–6 |
| 4 — Tech & equipment | ~25 entries | ~150 | 10–15 |
| 5 — Illustration | infra + batch | — | 3–4 |
| 6 — Slide deck refresh | deck update | — | 1–2 |
| **Total** | **~88 entries** | **~460 pp** | **27–40** |

Plus the existing ~24 entries from prior turns = ~112 entries,
~580 pages of campaign worldbuilding total.

## Decision points needing user input

1. **Tech-guide tone.** Should weapon/military entries lean
   toward *thriller-flavored realism* (Delta Green's register)
   or *Expanse-flavored hardware-spec* (the books'
   technical-but-readable register)? Default: Expanse-flavored.
2. **Cyberpunk-tech inclusion.** Cyberpunk-flavored material
   (black-market implants, off-record networks) is a tonal
   shift from the campaign's hard-SF/cosmic-horror baseline.
   Include but quarantine to a specific "underground" entry?
   Default: yes, quarantined.
3. **SDXL vs FLUX.** FLUX-dev produces higher-quality
   atmospheric art but is slower and consumes more VRAM.
   SDXL is a workhorse. Default: SDXL for batch, FLUX for
   hero pieces.
4. **Image style consistency.** Should all SDXL renders share
   a style anchor (e.g. *"oil painting, Simon Stalenhag,
   muted palette"*) or vary by entry mood? Default: style
   anchor with per-entry mood adjustments.

## Order of operations

This turn (Phase 1) executes immediately. Phase 2 onward
proceeds at user direction; the natural cadence is one phase
per session-day, but the user can re-prioritize at any point
(e.g. *"skip ahead to the medical-tech entry"* is fine).

The plan is a contract about scope, not a hard schedule.
