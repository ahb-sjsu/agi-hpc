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

from . import events
from .validator import ValidationResult, extract_code, validate
from .vmoe import Response, vMOE, default_experts

log = logging.getLogger("primer")

# Parallel lifecycle stream for the dashboard / evaluation pipeline.
# Wrapped in try so the service still runs if the module can't be
# imported (dev environments without the common package installed).
try:
    from agi.common.structured_log import LifecycleLogger

    _lifecycle = LifecycleLogger("primer")
except Exception:
    _lifecycle = None


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


def _stripped_task(task: dict) -> dict:
    """Shrink a raw task to just what the Primer needs.

    Drops the ``arc-gen`` augmentation block (hundreds of examples,
    100K+ tokens on some tasks — see service.py smoke postmortem) and
    caps the visible test list. We only verify against ``train``;
    ``arc-gen`` is for scaling up training, not for teaching the rule.
    """
    return {
        "train": task.get("train", []),
        "test": (task.get("test") or [])[:3],
    }


def _context_for_task(task_num: int, cfg: Config, task: dict) -> str:
    """Build the user-message context bundle: stripped task JSON +
    attempt history + any wiki articles Erebus has for this task."""
    parts: list[str] = []
    parts.append(
        f"## Task {task_num:03d}\n\n```json\n"
        f"{json.dumps(_stripped_task(task), indent=2)}\n```\n"
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

    # Relevant wiki articles — verified only; drafts would mislead the
    # proposer the same way they mislead Erebus (see task 381 incident).
    # Source is switched by EREBUS_CONTEXT_READER (wiki|graph, default wiki).
    snips = _context_snippets(task_num, cfg)
    if snips:
        parts.append("## Wiki context (what Erebus has already been taught)\n")
        parts.extend(snips)

    parts.append(
        "\n## Your task\n"
        "Write a verified sensei note following the output format in the system prompt."
    )
    return "\n".join(parts)


def _wiki_context_snippets(task_num: int, cfg: Config) -> list[str]:
    """Legacy retrieval: glob the wiki dir for verified task + meta notes.

    Reads only notes whose frontmatter carries a ``verified_by:`` field
    (the Primer's loader-side safety rule). Task 381 incident: an
    unverified draft note misled the ensemble, so this filter is load-
    bearing.
    """
    from agi.common.sensei_note import read_if_verified

    out: list[str] = []
    task_note = cfg.wiki_dir / f"sensei_task_{task_num:03d}.md"
    body = read_if_verified(task_note)
    if body is not None:
        out.append(f"### Existing sensei_task_{task_num:03d}.md\n{body[:4000]}")
    for meta in sorted(cfg.wiki_dir.glob("sensei_meta_*.md")):
        meta_body = read_if_verified(meta)
        if meta_body is not None:
            out.append(f"### {meta.name}\n{meta_body[:3000]}")
    return out


def _graph_context_snippets(task_num: int, cfg: Config) -> list[str]:
    """Graph-backed retrieval: query the UKG for eligible nodes and read
    their body files.

    Eligibility gate is centralized in
    ``agi.knowledge.graph.is_context_eligible``: a node must be
    ``filled ∧ verified ∧ active`` and its ``body_ref`` must exist on
    disk. Gap/stub/unverified nodes are visible to dashboards but are
    never fed to the generator as truth.

    Logs a warning if the graph returns zero eligible snippets for this
    task — that's the signal that the graph is empty, the wiki backfill
    hasn't run, or the body files are missing. Without this the graph
    reader would silently return no context and make the Primer look
    broken.
    """
    from agi.knowledge.graph import get_node, is_context_eligible, query_nodes

    out: list[str] = []
    task_id = f"sensei_task_{task_num:03d}"
    task_node = get_node(task_id)
    if task_node is not None and is_context_eligible(task_node, wiki_root=cfg.wiki_dir):
        try:
            body = (cfg.wiki_dir / task_node["body_ref"]).read_text(encoding="utf-8")
            out.append(f"### Existing {task_node['body_ref']}\n{body[:4000]}")
        except OSError as e:
            log.warning("graph_context: cannot read %s: %s", task_node["body_ref"], e)

    # Meta notes — cross-cutting wisdom, always included when eligible.
    for meta in query_nodes(type="filled", verified=True, status="active"):
        nid = meta.get("id") or ""
        if not nid.startswith("sensei_meta_"):
            continue
        if nid == task_id:  # already included above
            continue
        if not is_context_eligible(meta, wiki_root=cfg.wiki_dir):
            continue
        try:
            body = (cfg.wiki_dir / meta["body_ref"]).read_text(encoding="utf-8")
        except OSError:
            continue
        out.append(f"### {meta['body_ref']}\n{body[:3000]}")

    if not out:
        log.warning(
            "graph_context: zero eligible snippets for task %s "
            "(graph empty, backfill missing, or body files absent?)",
            task_id,
        )
    return out


def _context_snippets(task_num: int, cfg: Config) -> list[str]:
    """Dispatch to wiki or graph retrieval per EREBUS_CONTEXT_READER.

    Default is ``wiki`` until graph-backed retrieval is validated in
    practice. Per the spec's rollout-safety note, no automatic wiki
    fallback on an empty graph result — the loud warning in
    ``_graph_context_snippets`` is sufficient for operators to notice
    and flip the flag back.
    """
    from agi.knowledge.graph import context_reader_mode

    mode = context_reader_mode()
    if mode == "graph":
        return _graph_context_snippets(task_num, cfg)
    return _wiki_context_snippets(task_num, cfg)


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


def _upsert_graph_node(task_num: int, parsed: dict, note_path: Path) -> None:
    """Upsert the knowledge-graph node for a freshly published sensei note.

    Fire-and-forget: graph writes must never fail the publish path.
    A gap node with the same id (seeded by the help-queue import in a
    later phase) is promoted to filled+verified with ``created_at``
    preserved — exactly the v1 spec's gap→filled transition.
    """
    try:
        from agi.knowledge.graph import normalize_tags, upsert_node

        cls = (parsed.get("class") or "TRANSFORMATION").lower()
        family = (parsed.get("family") or "").strip() or ""
        topic = family or cls
        tag_list = normalize_tags([cls, family, "arc", "primer"])
        pretty = (family or cls).replace("-", " ").replace("_", " ").strip() or "task"
        title = f"Task {task_num:03d} — {pretty}"
        upsert_node(
            id=f"sensei_task_{task_num:03d}",
            type="filled",
            status="active",
            topic=topic,
            tags=tag_list,
            title=title,
            body_ref=note_path.name,
            verified=True,
            source="primer",
            evidence=[f"primer_task:{task_num:03d}"],
        )
    except Exception as e:  # noqa: BLE001 — never fail the publish path
        log.warning("graph upsert failed for task %s: %s", task_num, e)


def _publish_note(task_num: int, parsed: dict, code: str, cfg: Config) -> Path:
    """Write the sensei note + graph upsert + git commit + push."""
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
    # Upsert knowledge-graph node alongside the wiki write so readers
    # see graph + file consistently. Errors are warnings only.
    _upsert_graph_node(task_num, parsed, path)
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
_HEALTH_STATE = Path("/archive/neurogolf/primer_health.json")


def _save_health(moe: vMOE) -> None:
    """Persist expert-health summary for dashboard consumption."""
    try:
        from agi.common.atomic_write import atomic_write_text

        atomic_write_text(_HEALTH_STATE, json.dumps(moe.health.summary(), indent=2))
    except Exception as e:
        log.warning("health save failed: %s", e)


def _load_cooldown() -> dict[str, float]:
    try:
        return json.loads(_COOLDOWN_STATE.read_text())
    except Exception:
        return {}


def _save_cooldown(state: dict[str, float]) -> None:
    try:
        from agi.common.atomic_write import atomic_write_text

        atomic_write_text(_COOLDOWN_STATE, json.dumps(state))
    except Exception as e:
        log.warning("cooldown save failed: %s", e)


_HELP_TASK_RE = re.compile(r"task\s*(\d+)", re.IGNORECASE)


def _pick_stuck_tasks(cfg: Config) -> list[int]:
    """Return task numbers to process this tick, ordered by priority.

    Priority heuristic (highest first):
      tier 1 — tasks with partial progress (``best_correct > 0``)
               AND at least ``min_attempts``. These are "almost-solved";
               one good nudge is most likely to land.
      tier 2 — tasks at 0 progress but with many attempts. Hardest,
               most likely to fail, but we still probe them because
               a fresh-perspective ensemble can crack genuinely novel
               rules that arc_scientist has been pattern-matching wrong.

    Within each tier we sort by descending attempt count (most-stuck
    first). Tasks that have been touched within ``cooldown_s`` are
    excluded regardless of tier."""
    cooldown = _load_cooldown()
    now = time.time()
    try:
        mem = json.loads(cfg.memory_path.read_text())
    except Exception:
        mem = {}
    tier1: list[tuple[int, int]] = []  # (task_num, attempts) — partial progress
    tier2: list[tuple[int, int]] = []  # all-zero-progress stuck
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
        best = tk.get("best_correct", 0) or 0
        if best > 0:
            tier1.append((tn, attempts))
        else:
            tier2.append((tn, attempts))
    tier1.sort(key=lambda p: -p[1])
    tier2.sort(key=lambda p: -p[1])
    return [tn for tn, _ in tier1] + [tn for tn, _ in tier2]


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
    # Drop experts currently in a health cooldown — the canary (qwen3)
    # being consistently slow means NRP's thinking models will definitely
    # timeout too; don't burn 5 min each on foregone conclusions.
    candidate = moe.healthy_subset(["kimi", "glm-4.7", "qwen3"])
    if not candidate:
        # All experts degraded — still probe qwen3 once to see if it
        # recovered. A probe succeeding clears the degradation via the
        # tracker's next ``record()`` call.
        candidate = ["qwen3"]
    log.info("task%03d: consulting vMOE ensemble (%s)", tn, " + ".join(candidate))
    responses = await moe.ensemble(
        messages,
        experts=candidate,
        max_tokens=12000,
        temperature=0.2,
        return_all=True,
    )
    if _lifecycle:
        _lifecycle.emit(
            "ensemble_complete",
            task=tn,
            experts=[r.expert for r in responses],
            latencies_s={r.expert: round(r.latency_s, 2) for r in responses},
            errors={r.expert: r.error for r in responses if not r.ok},
        )
    for r in responses:
        if not r.ok:
            log.info("task%03d: %s failed (%s)", tn, r.expert, r.error)
            events.append(
                task=tn,
                expert=r.expert,
                ok=False,
                latency_s=r.latency_s,
                verify_pass=False,
                published=False,
                error=r.error,
            )
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
            if _lifecycle:
                _lifecycle.emit(
                    "publish",
                    task=tn,
                    expert=r.expert,
                    latency_s=round(r.latency_s, 2),
                    family=(parsed.get("family") or ""),
                    note_path=path.name,
                )
            events.append(
                task=tn,
                expert=r.expert,
                ok=True,
                latency_s=r.latency_s,
                verify_pass=True,
                published=True,
            )
            return True
        log.info("task%03d: %s did not verify: %s", tn, r.expert, vr.diagnostic[:200])
        # Log a peek at the raw response so we can see WHY extraction
        # or validation failed (missing code block, wrong format,
        # truncated output, etc).
        peek = (r.content or "").replace("\n", " \u23ce ")[:400]
        log.info("task%03d: %s raw-peek: %s", tn, r.expert, peek)
        if _lifecycle:
            _lifecycle.emit(
                "verify_fail",
                task=tn,
                expert=r.expert,
                latency_s=round(r.latency_s, 2),
                diagnostic=vr.diagnostic[:200],
            )
        events.append(
            task=tn,
            expert=r.expert,
            ok=True,
            latency_s=r.latency_s,
            verify_pass=False,
            published=False,
            error=vr.diagnostic[:200],
        )
    log.info("task%03d: no expert produced a verified solution this round", tn)
    return False


def _refresh_gaps_from_help_queue(cfg: Config) -> None:
    """Sync UKG gap nodes with Erebus's help queue. Fire-and-forget.

    Called at the top of every tick so new stuck-task questions appear
    as gaps without waiting for a daemon restart. Idempotent — skips
    entries that are already filled or whose gap is up-to-date.
    """
    try:
        from agi.knowledge.gap_import import import_help_queue

        rep = import_help_queue(cfg.help_path)
        if rep.imported or rep.refreshed or rep.failed:
            log.info("gap_sync: %s", rep.summary())
    except Exception as e:  # noqa: BLE001 — never fail the tick
        log.warning("gap_sync_failed: %s", e)


async def tick(cfg: Config, moe: vMOE) -> int:
    """One polling tick: process as many stuck tasks as verify-publish
    per tick (capped at 3 to avoid runaway wiki commits)."""
    _refresh_gaps_from_help_queue(cfg)
    picks = _pick_stuck_tasks(cfg)
    if not picks:
        log.info("tick: no stuck tasks ready to process")
        if _lifecycle:
            _lifecycle.emit("tick_empty", candidates=0)
        return 0
    log.info("tick: %d candidate stuck tasks; processing top 3", len(picks))
    if _lifecycle:
        _lifecycle.emit(
            "tick_start",
            candidates=len(picks),
            will_process=min(3, len(picks)),
            picks=picks[:3],
        )
    cooldown = _load_cooldown()
    published = 0
    for tn in picks[:3]:
        ok = await _process_one(tn, cfg, moe)
        cooldown[str(tn)] = time.time()  # set cooldown regardless of outcome
        _save_cooldown(cooldown)
        _save_health(moe)
        if ok:
            published += 1
    if _lifecycle:
        _lifecycle.emit(
            "tick_complete",
            candidates=len(picks),
            processed=min(3, len(picks)),
            published=published,
        )
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
