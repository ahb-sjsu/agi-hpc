# Handout sources

Each `.md` file in this directory is a player handout. The file name
is arbitrary; the **slug** in the front-matter is what the renderer
uses as the output filename.

## Front-matter contract

Every handout starts with:

```yaml
---
slug: <output-filename-without-extension>
title: "Human-readable title"
audience: all | player:<id>
secrets: none | <comma-separated markers>
---
```

- `audience=all` ⇒ generic handout shared with every player.
- `audience=player:<id>` ⇒ per-character, gated behind that player's
  identity in the GM portal.
- `secrets=none` ⇒ safe to share in any channel. Any other value is a
  free-form marker (`mi-go`, `keeper-only`, `session1-spoilers`) used
  by the portal to filter what's visible in crew chat.

## Rendering

```bash
# Render everything to /tmp/handouts/
python scripts/generate_handouts.py --out /tmp/handouts

# Render one handout
python scripts/generate_handouts.py --slug session0_briefing --out dist/
```

Requires `pandoc` + `xelatex` on the host. Atlas has both; dev boxes
may need `apt install pandoc texlive-xetex`.

## Adding a handout

1. Drop a new `.md` file here with a valid front-matter block.
2. Prose in GitHub-flavored Markdown.
3. Commit the source; the PDF is generated at deploy time, not
   committed.
4. If it's per-character, set `audience: player:<id>` matching the
   character's sheet row.
