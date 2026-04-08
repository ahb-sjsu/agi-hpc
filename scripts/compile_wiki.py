#!/usr/bin/env python3
"""Compile a structured wiki from AGI-HPC code repos using Spock (Gemma 4).

Karpathy-style "LLM Knowledge Base": the LLM reads each repo/subsystem
and writes structured markdown articles with backlinks, summaries, and
key concepts. The wiki is the Tier 1 search layer — instant, exact
answers to "how does X work?" questions.

Usage:
    python compile_wiki.py                    # compile all repos
    python compile_wiki.py --repo agi-hpc     # compile one repo
    python compile_wiki.py --lint             # lint existing wiki
    python compile_wiki.py --list             # list articles

Articles are written to /archive/wiki/ (configurable).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wiki-compiler")

WIKI_DIR = Path(os.environ.get("WIKI_DIR", "/archive/wiki"))
REPOS_DIR = Path("/archive/ahb-sjsu")
LLM_URL = "http://localhost:8080"
MAX_CONTEXT = 6000  # chars to send to LLM per article

# Subsystems to document for agi-hpc
AGI_HPC_SUBSYSTEMS = [
    ("safety-gateway", "src/agi/safety/", "The Safety Gateway (DEME) subsystem"),
    ("event-fabric", "src/agi/core/events/", "The NATS Event Fabric"),
    ("semantic-memory", "src/agi/memory/semantic/", "Semantic Memory (RAG + pgvector)"),
    ("episodic-memory", "src/agi/memory/episodic/", "Episodic Memory"),
    ("procedural-memory", "src/agi/memory/procedural/", "Procedural Memory"),
    ("metacognition", "src/agi/metacognition/", "Metacognition subsystem"),
    ("left-hemisphere", "src/agi/lh/", "Left Hemisphere (analytical reasoning)"),
    ("right-hemisphere", "src/agi/rh/", "Right Hemisphere (creative/pattern)"),
    ("llm-inference", "src/agi/meta/llm/", "LLM Inference layer (TurboQuant, client)"),
    ("hybrid-search", "src/agi/common/hybrid_search.py", "Hybrid Search (RRF, wiki, vector+FTS)"),
    ("embedding-compression", "src/agi/common/embedding_codec.py", "Embedding Compression (TurboQuant codec)"),
    ("integration", "src/agi/integration/", "Subsystem Integration layer"),
    ("environment", "src/agi/env/", "Environment Interface"),
]


def read_source_files(repo_path: Path, subdir: str, max_chars: int = MAX_CONTEXT) -> str:
    """Read source files from a directory, concatenated and truncated."""
    target = repo_path / subdir
    if target.is_file():
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]

    if not target.is_dir():
        return ""

    content = []
    total = 0
    for py_file in sorted(target.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        header = f"\n--- {py_file.relative_to(repo_path)} ---\n"
        if total + len(text) + len(header) > max_chars:
            remaining = max_chars - total - len(header)
            if remaining > 100:
                content.append(header + text[:remaining] + "\n[truncated]")
            break
        content.append(header + text)
        total += len(text) + len(header)

    return "\n".join(content)


def compile_article(
    slug: str,
    source_code: str,
    description: str,
    related_slugs: list[str],
) -> str:
    """Ask Spock to write a wiki article from source code."""
    backlinks = ", ".join(f"[[{s}]]" for s in related_slugs)
    prompt = (
        f"You are writing a technical wiki article about: {description}\n\n"
        f"Write a concise, structured Markdown article with:\n"
        f"1. A # heading with the component name\n"
        f"2. A 2-3 sentence overview\n"
        f"3. ## Architecture section explaining the design\n"
        f"4. ## Key Classes/Functions with brief descriptions\n"
        f"5. ## Configuration (if any config files/params exist)\n"
        f"6. ## Related: {backlinks}\n\n"
        f"Base your article ONLY on this source code:\n\n"
        f"```\n{source_code}\n```\n\n"
        f"Write the article now. Be factual — only describe what the code actually does."
    )

    try:
        resp = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error("LLM call failed for %s: %s", slug, e)

    return f"# {description}\n\n*Article compilation failed. Source available in repo.*\n"


def lint_article(slug: str, content: str) -> list[str]:
    """Check a wiki article for issues."""
    issues = []
    if not content.startswith("# "):
        issues.append(f"{slug}: missing # heading")
    if "## " not in content:
        issues.append(f"{slug}: no subsections")
    if "[[" not in content:
        issues.append(f"{slug}: no backlinks")
    if len(content) < 200:
        issues.append(f"{slug}: suspiciously short ({len(content)} chars)")
    return issues


def compile_all(repo_filter: str | None = None):
    """Compile wiki articles for all AGI-HPC subsystems."""
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    repo_path = REPOS_DIR / "agi-hpc"
    if not repo_path.exists():
        repo_path = Path("/home/claude/agi-hpc")

    all_slugs = [s[0] for s in AGI_HPC_SUBSYSTEMS]
    compiled = 0

    for slug, subdir, description in AGI_HPC_SUBSYSTEMS:
        if repo_filter and slug != repo_filter:
            continue

        article_path = WIKI_DIR / f"{slug}.md"
        if article_path.exists():
            log.info("[wiki] %s already exists, skipping (use --force to overwrite)", slug)
            continue

        log.info("[wiki] compiling %s from %s...", slug, subdir)
        source = read_source_files(repo_path, subdir)
        if not source:
            log.warning("[wiki] no source found for %s at %s", slug, subdir)
            continue

        related = [s for s in all_slugs if s != slug][:5]
        article = compile_article(slug, source, description, related)

        article_path.write_text(article, encoding="utf-8")
        compiled += 1
        log.info("[wiki] wrote %s (%d chars)", article_path, len(article))

        time.sleep(1)  # rate limit

    # Write index
    index_path = WIKI_DIR / "index.md"
    index_lines = ["# AGI-HPC Wiki\n", f"*{len(all_slugs)} articles*\n"]
    for slug, _, desc in AGI_HPC_SUBSYSTEMS:
        exists = (WIKI_DIR / f"{slug}.md").exists()
        status = "+" if exists else "-"
        index_lines.append(f"- [{status}] [[{slug}]] — {desc}")
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    log.info("[wiki] compiled %d articles, index at %s", compiled, index_path)


def lint_all():
    """Lint all wiki articles."""
    if not WIKI_DIR.exists():
        print("No wiki directory found at %s" % WIKI_DIR)
        return

    all_issues = []
    for md_file in sorted(WIKI_DIR.glob("*.md")):
        if md_file.name == "index.md":
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        issues = lint_article(md_file.stem, content)
        all_issues.extend(issues)

    if all_issues:
        print("Wiki lint issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("Wiki lint: all articles pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile AGI-HPC wiki")
    parser.add_argument("--repo", type=str, default=None, help="Compile specific article")
    parser.add_argument("--lint", action="store_true", help="Lint existing articles")
    parser.add_argument("--list", action="store_true", help="List articles")
    parser.add_argument("--force", action="store_true", help="Overwrite existing articles")
    args = parser.parse_args()

    if args.lint:
        lint_all()
    elif args.list:
        if WIKI_DIR.exists():
            for md in sorted(WIKI_DIR.glob("*.md")):
                print(f"  {md.stem} ({md.stat().st_size} bytes)")
        else:
            print("No wiki at %s" % WIKI_DIR)
    else:
        compile_all(args.repo)
