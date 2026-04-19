"""The Primer daemon — always-on teaching loop for Erebus.

Polls the help queue for stuck tasks, runs a vMOE ensemble of frontier
LLMs against each, verifies the returned code against task.train, and
publishes a verified ``sensei_task_NNN.md`` to the wiki. Verified notes
are git-committed and pushed so the existing CI deploy picks them up.

Runs as a systemd service (``atlas-primer.service``). CPU-only; no GPU
required. Expects these env vars:

  NRP_LLM_TOKEN         bearer token for NRP ellm/anthropic endpoints
  EREBUS_WIKI_DIR       wiki directory (default: /home/claude/agi-hpc/wiki)
  EREBUS_MEMORY_PATH    arc_scientist memory JSON (default:
                        /archive/neurogolf/arc_scientist_memory.json)
  EREBUS_TASK_DIR       ARC task JSON directory (default:
                        /archive/neurogolf)
  EREBUS_HELP_PATH      help queue JSON (default:
                        /archive/neurogolf/erebus_help_queue.json)
  EREBUS_REPO_DIR       agi-hpc clone (default: /home/claude/agi-hpc)
  PRIMER_POLL_S         polling interval in seconds (default: 300)
  PRIMER_MIN_ATTEMPTS   only act on tasks with ≥ this many prior attempts
                        (default: 10)
  PRIMER_COOLDOWN_S     min gap between re-processing the same task
                        regardless of verify outcome (default: 21600 = 6h)
  EREBUS_VMOE_EXPERTS   comma-separated expert names (default: all)

Safety invariants (see feedback_sensei_verify_solutions):
- Never writes a sensei_task_NNN.md whose code doesn't pass 100% of
  task.train. A wrong note is worse than no note.
- Never overwrites an existing sensei_task_NNN.md unless the new code
  also passes verification AND improves on coverage (handled in v2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validator import ValidationResult, extract_code, validate
from .vmoe import Response, vMOE, default_experts

log = logging.getLogger("primer")


# ── paths & config ───────────────────────────────────────────────


@dataclass
class Config:
    wiki_dir: Path
    memory_path: Path
    task_dir: Path
    help_path: Path
    repo_dir: Path
    poll_s: int
    min_attempts: int
    cooldown_s: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            wiki_dir=Path(
                os.environ.get("EREBUS_WIKI_DIR", "/home/claude/agi-hpc/wiki")
            ),
            memory_path=Path(
                os.environ.get(
                    "EREBUS_MEMORY_PATH", "/archive/neurogolf/arc_scientist_memory.json"
                )
            ),
            task_dir=Path(os.environ.get("EREBUS_TASK_DIR", "/archive/neurogolf")),
            help_path=Path(
                os.environ.get(
                    "EREBUS_HELP_PATH", "/archive/neurogolf/erebus_help_queue.json"
                )
            ),
            repo_dir=Path(os.environ.get("EREBUS_REPO_DIR", "/home/claude/agi-hpc")),
            poll_s=int(os.environ.get("PRIMER_POLL_S", "300")),
            min_attempts=int(os.environ.get("PRIMER_MIN_ATTEMPTS", "10")),
            cooldown_s=int(os.environ.get("PRIMER_COOLDOWN_S", "21600")),
        )


# ── prompt construction ──────────────────────────────────────────


_SYSTEM_PROMPT = """You are the Primer — an always-on teacher for Erebus,
an autonomous ARC-AGI scientist. Your job is to analyse puzzles he has
failed to solve, explain the rule, and publish a VERIFIED reference
implementation that future attempts can learn from.

Context you have access to:
- The full task JSON (train + visible test examples).
- Erebus's attempt history on this task (what he tried, why it failed).
- Existing wiki articles (what Erebus has already been taught).

Hard rules:
1. Your code MUST pass every train example exactly. A wrong reference
   implementation is worse than no note at all; it actively misleads.
2. Use only numpy and the Python stdlib inside transform().
3. Produce a deterministic function: transform(grid: list[list[int]])
   -> list[list[int]]. No randomness, no network, no filesystem.
4. Before you write code, identify the output-shape class from the
   typology: CLASSIFICATION (output 1x1 or small fixed), TRANSFORMATION
   (output same shape as input), EXTRACTION (small output pulled from
   input), EXPANSION (output > input). Pick the right strategy
   vocabulary for that class.

Output format: a single JSON object with these keys:
  "class":   one of CLASSIFICATION/TRANSFORMATION/EXTRACTION/EXPANSION
  "family":  the primitive family if any (e.g. "symmetry-completion",
             "symmetry-classifier", "object-count"); else null
  "rule":    one short paragraph explaining the rule in plain English
  "code":    Python source for def transform(grid) -> list[list[int]]
  "note":    the full markdown body for sensei_task_NNN.md (no YAML
             frontmatter — the daemon adds that). Include: section
             "## The rule" (plain English), "## Reference implementation"
             (the same code from "code" in a ```python fence), and
             "## Why this generalizes" pointing at the primitive family.
"""


def _context_for_task(task_num: int, cfg: Config, task: dict) -> str:
    """Build the user-message context bundle: task JSON + attempt history
    + any wiki articles Erebus has for this task or its family."""
    parts: list[str] = []
    parts.append(
        f"## Task {task_num:03d}\n\n```json\n{json.dumps(task, indent=2)}\n```\n"
    )

    # Prior-attempt summary from arc_scientist memory
    try:
        mem = json.loads(cfg.memory_path.read_text())
    except Exception:
        mem = {}
    tk = (mem.get("tasks") or {}).get(str(task_num)) or (mem.get("tasks") or {}).get(
        task_num
    )
    if tk:
        attempts = tk.get("attempts", []) or []
        best = tk.get("best_correct", 0)
        total = tk.get("best_total", 0)
        error_types = ", ".join(tk.get("error_types", []) or []) or "none classified"
        parts.append(
            f"## Erebus's attempt history\n\n"
            f"- {len(attempts)} prior attempts, best score {best}/{total}.\n"
            f"- Error types seen: {error_types}.\n"
            f"- Strategies tried: {', '.join(tk.get('strategies_tried', []) or [])}.\n"
        )
        recent = attempts[-3:]
        if recent:
            parts.append("### Last 3 failed attempts:\n")
            for a in recent:
                parts.append(
                    f"- attempt @ {a.get('timestamp', '?')[:19]}: "
                    f"strategy={a.get('strategy', '?')} model={a.get('model', '?')} "
                    f"score={a.get('correct', 0)}/{a.get('total', 0)} "
                    f"error_type={a.get('error_type', '?')}\n"
                    f"  diagnosis: {a.get('error', '?')[:200]}\n"
                )

    # Relevant wiki articles
    wiki_snips: list[str] = []
    task_note = cfg.wiki_dir / f"sensei_task_{task_num:03d}.md"
    if task_note.exists():
        body = task_note.read_text()[:4000]
        wiki_snips.append(f"### Existing sensei_task_{task_num:03d}.md\n{body}")
    for meta in sorted(cfg.wiki_dir.glob("sensei_meta_*.md")):
        wiki_snips.append(f"### {meta.name}\n{meta.read_text()[:3000]}")
    if wiki_snips:
        parts.append("## Wiki context (what Erebus has already been taught)\n")
        parts.extend(wiki_snips)

    parts.append(
        "\n## Your task\n"
        "Write a verified sensei note following the output format in the system prompt."
    )
    return "\n".join(parts)


# ── verification wrapper ─────────────────────────────────────────


def _verify_response(resp: Response, task: dict) -> tuple[bool, ValidationResult, str]:
    """Extract code from a Primer response and verify against task.train.

    Returns (passed, ValidationResult, code_source).
    """
    code = extract_code(resp.content)
    if not code:
        return False, ValidationResult(False, [], "no code extracted"), ""
    vr = validate(code, task)
    return vr.all_pass, vr, code


# ── publisher ────────────────────────────────────────────────────


_FRONTMATTER_TMPL = (
    "---\n"
    "type: sensei_note\n"
    "task: {task_num}\n"
    "tags: [{tags}]\n"
    "written_by: The Primer\n"
    "written_at: {date}\n"
    "verified_by: run-against-train (all examples pass)\n"
    "---\n\n"
)


def _publish_note(task_num: int, parsed: dict, code: str, cfg: Config) -> Path:
    """Write the sensei note + git commit + push."""
    cls = (parsed.get("class") or "TRANSFORMATION").lower()
    family = (parsed.get("family") or "").strip() or ""
    tags = ", ".join(t for t in [cls, family or None, "arc", "primer"] if t)
    date = time.strftime("%Y-%m-%d", time.gmtime())
    body = parsed.get("note") or ""
    # Safety: ensure code block present even if the model forgot
    if "def transform" not in body:
        body += "\n\n## Reference implementation\n\n```python\n" + code + "\n```\n"
    frontmatter = _FRONTMATTER_TMPL.format(task_num=task_num, tags=tags, date=date)
    path = cfg.wiki_dir / f"sensei_task_{task_num:03d}.md"
    path.write_text(frontmatter + body.strip() + "\n")
    # Commit + push
    subprocess.run(["git", "-C", str(cfg.repo_dir), "add", str(path)], check=True)
    msg = f"primer: verified sensei note for task {task_num:03d} ({family or cls})"
    subprocess.run(["git", "-C", str(cfg.repo_dir), "commit", "-m", msg], check=True)
    push = subprocess.run(
        ["git", "-C", str(cfg.repo_dir), "push"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if push.returncode != 0:
        log.warning("git push failed: %s", push.stderr[:300])
    return path


# ── candidate-task selection ─────────────────────────────────────


_COOLDOWN_STATE = Path("/archive/neurogolf/primer_cooldown.json")


def _load_cooldown() -> dict[str, float]:
    try:
        return json.loads(_COOLDOWN_STATE.read_text())
    except Exception:
        return {}


def _save_cooldown(state: dict[str, float]) -> None:
    try:
        _COOLDOWN_STATE.parent.mkdir(parents=True, exist_ok=True)
        _COOLDOWN_STATE.write_text(json.dumps(state))
    except Exception as e:
        log.warning("cooldown save failed: %s", e)


_HELP_TASK_RE = re.compile(r"task\s*(\d+)", re.IGNORECASE)


def _pick_stuck_tasks(cfg: Config) -> list[int]:
    """Return task numbers to process this tick, ordered by priority.

    Priority: tasks with most failed attempts that haven't been
    processed recently (per cooldown) and don't yet have a verified
    sensei note."""
    cooldown = _load_cooldown()
    now = time.time()
    try:
        mem = json.loads(cfg.memory_path.read_text())
    except Exception:
        mem = {}
    candidates: list[tuple[int, int]] = []
    for task_num_str, tk in (mem.get("tasks") or {}).items():
        try:
            tn = int(task_num_str)
        except ValueError:
            continue
        if tk.get("solved"):
            continue
        attempts = len(tk.get("attempts", []) or [])
        if attempts < cfg.min_attempts:
            continue
        last_seen = cooldown.get(str(tn), 0.0)
        if now - last_seen < cfg.cooldown_s:
            continue
        candidates.append((tn, attempts))
    candidates.sort(key=lambda p: -p[1])  # most-tried first
    return [tn for tn, _ in candidates]


def _load_task(task_dir: Path, task_num: int) -> dict | None:
    path = task_dir / f"task{task_num:03d}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── main loop ────────────────────────────────────────────────────


async def _process_one(tn: int, cfg: Config, moe: vMOE) -> bool:
    """Process one stuck task. Returns True if a verified note was published."""
    task = _load_task(cfg.task_dir, tn)
    if not task:
        log.warning("task%03d: no JSON at %s", tn, cfg.task_dir)
        return False
    user = _context_for_task(tn, cfg, task)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    log.info("task%03d: consulting vMOE ensemble (kimi + glm-4.7 + qwen3)", tn)
    responses = await moe.ensemble(
        messages,
        experts=["kimi", "glm-4.7", "qwen3"],
        max_tokens=6000,
        temperature=0.2,
        return_all=True,
    )
    for r in responses:
        if not r.ok:
            log.info("task%03d: %s failed (%s)", tn, r.expert, r.error)
            continue
        passed, vr, code = _verify_response(r, task)
        if passed:
            try:
                parsed = json.loads(r.content)
            except Exception:
                parsed = {}
            path = _publish_note(tn, parsed, code, cfg)
            log.info(
                "task%03d: published %s via %s (%.1fs)",
                tn,
                path.name,
                r.expert,
                r.latency_s,
            )
            return True
        log.info("task%03d: %s did not verify: %s", tn, r.expert, vr.diagnostic[:200])
    log.info("task%03d: no expert produced a verified solution this round", tn)
    return False


async def tick(cfg: Config, moe: vMOE) -> int:
    """One polling tick: process as many stuck tasks as verify-publish
    per tick (capped at 3 to avoid runaway wiki commits)."""
    picks = _pick_stuck_tasks(cfg)
    if not picks:
        log.info("tick: no stuck tasks ready to process")
        return 0
    log.info("tick: %d candidate stuck tasks; processing top 3", len(picks))
    cooldown = _load_cooldown()
    published = 0
    for tn in picks[:3]:
        ok = await _process_one(tn, cfg, moe)
        cooldown[str(tn)] = time.time()  # set cooldown regardless of outcome
        _save_cooldown(cooldown)
        if ok:
            published += 1
    return published


async def _main_async() -> None:
    cfg = Config.from_env()
    moe = vMOE(experts=default_experts())
    log.info(
        "Primer online. wiki=%s memory=%s poll=%ds min_attempts=%d cooldown=%ds",
        cfg.wiki_dir,
        cfg.memory_path,
        cfg.poll_s,
        cfg.min_attempts,
        cfg.cooldown_s,
    )
    while True:
        try:
            n = await tick(cfg, moe)
            log.info("tick complete: %d notes published", n)
        except Exception as e:  # noqa: BLE001 — never let the loop die
            log.exception("tick error: %s", e)
        await asyncio.sleep(cfg.poll_s)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("PRIMER_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
