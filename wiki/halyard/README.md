# Halyard Table — Wiki / RAG Corpus

Source-of-truth markdown for the Beyond the Heliopause setting.
`scripts/halyard/build_bible.py` walks this tree and produces
`/archive/halyard/bible/halyard_bible.json`, which the Halyard
Keeper's LLM path loads as the `Bible` passed into
`agi.primer.artemis.mode.handle_turn`.

## Directory layout

```
wiki/halyard/
├── README.md                    — this file
├── setting/                     — solar system, 2348 context
├── factions/                    — UN, MCR, OPA, M-K, Protogen
├── ship/                        — MKS Halyard hull, systems, history
├── crew/                        — the standing crew (one file per NPC)
├── locations/                   — Ceres, Nithon (public), Luna, Mars
├── tech/                        — Epstein drive, MCALs, AI in 2348
├── mission/                     — the contract, the charter
└── mythos/                      — spoiler-tagged, not surfaced to AIs
```

## Frontmatter

Every `.md` file starts with YAML frontmatter declaring:

```yaml
---
id:            short-stable-identifier
title:         Human-readable page title
artemis:       known | unknown | forbidden
sigma4:        known | unknown | forbidden
topic:         setting | faction | ship | crew | location | tech | mission | mythos
tags:          [optional, extra, hash-tag-ish, labels]
---
```

`known`: chunk is added to this AI's context and is safe to quote.
`unknown`: chunk is *not* added to context; validator will also
reject replies that n-gram-overlap the chunk — defense in depth.
`forbidden`: as unknown, plus any exact phrase listed under
`forbidden_phrases` in the frontmatter is added to the hard
reject list.

A directory's default visibility may be overridden per-file.
`mythos/` defaults to `forbidden` for both AIs. `setting/`,
`factions/`, `ship/`, `crew/`, `locations/`, `tech/`, `mission/`
default to `known` for both unless the file overrides.

## Build

```bash
python scripts/halyard/build_bible.py
# → writes /archive/halyard/bible/halyard_bible.json
```

halyard-keeper reloads the bible on SIGHUP or on next process
start. No stop needed.

## Writing rules

- **Stay in-universe.** Present tense, factual tone, no
  fourth-wall meta.
- **Date everything that moves.** Populations, fleet sizes,
  construction dates. The AIs answer better with grounded
  numbers.
- **One topic per file.** Split rather than merge.
- **Keep it tight.** 200–500 words per page is the sweet spot;
  longer pages fragment retrieval relevance.
- **Cross-reference by id**, not file path: `see also: [crew/marsh]`.
