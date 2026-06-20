# Contributing — New-Contributor Onboarding

Welcome! Work through this top-to-bottom for your first task. If anything's
unclear, ask **in your issue or PR** (not DMs) — questions are expected and
encouraged.

## 1 · Access
- [ ] Send your **GitHub username** to the maintainer.
- [ ] **Accept the repo invite** (email / github.com/notifications) — this lets
      your issue be assigned to you.
- [ ] These repos are public, so you can also just **fork** and work from your fork.

## 2 · Local setup
- [ ] Install **Python 3.12**.
- [ ] Clone your fork (or the repo).
- [ ] Create + activate a virtualenv: `python -m venv .venv`
- [ ] Install dev deps: **`pip install -e ".[test,dev]"`**
- [ ] **Sanity-check before changing anything** — the code lives under `src/`:
      `PYTHONPATH=src pytest tests/ -q`

## 3 · Your task
- [ ] Find the **`good first issue`** assigned to you; read it fully, including
      the **Pointers** section.
- [ ] Open the referenced files and **reproduce the current behavior first** —
      understand before you change.
- [ ] Stay in scope: do what the issue asks; note other ideas in a comment.

## 4 · Branch & build
- [ ] Branch off `main`: **`feat/<short-name>`** or `fix/<short-name>`.
- [ ] **Small, frequent commits** with clear messages.
- [ ] Match the surrounding code's style (naming, structure, comment density).
- [ ] **Never commit** secrets/tokens, virtualenvs, large binaries, or generated files.

## 5 · Quality gates (green *before* you ask for review)
Run locally — CI enforces the same on every push:
- [ ] **Lint:** `ruff check src/ tests/ --select F --ignore F401`
- [ ] **Format** *(only where the repo's CI runs black):* `black --check src/ tests/`
      — run `black src/ tests/` to fix.
- [ ] **Tests:** `PYTHONPATH=src pytest tests/ -q` all pass.

## 6 · Open a **draft** PR early
- [ ] Open as a **Draft PR** as soon as something runs — don't wait for "done."
- [ ] Put **`Closes #<issue>`** in the description.
- [ ] Push often; **CI runs on every push** — keep it green.
- [ ] Mark "Ready for review" + request the maintainer when it's ready.

## 7 · Review → merge
- [ ] Address feedback with **new commits** (don't force-push mid-review unless asked).
- [ ] Keep the PR **scoped to the one issue**.
- [ ] The maintainer **squash-merges** once CI is green and it's approved.

## 8 · Good habits
- [ ] Stuck for more than ~30 min? Post what you tried in the PR/issue and ask.
- [ ] Flag blockers early — a half-working draft PR beats silence.
- [ ] Be kind in reviews; assume good intent.

---
*This is a per-task checklist — copy the relevant boxes into your PR description
and tick them off as you go.*
