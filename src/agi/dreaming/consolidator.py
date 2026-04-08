# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Memory Consolidator — Episodic Replay to Wiki Article Synthesis.

The core of the dreaming subsystem. Reads recent episodic memories
(conversations, interactions), clusters them by topic, and uses the
LLM (Superego) to synthesize structured wiki articles.

Output articles follow the Karpathy wiki pattern:
  - Title and summary
  - Key concepts with definitions
  - Detailed explanation with examples drawn from actual episodes
  - Backlinks to related articles
  - Provenance: which episodes contributed to this knowledge
  - Confidence level and last-verified timestamp

The wiki is Tier 1 in the RAG search cascade (<1ms lookup, no embedding).
Every dream cycle potentially creates or updates articles, making Atlas
smarter by morning.

Biological analogy:
  - Hippocampal replay: re-reading episodic memories
  - Pattern extraction: LLM identifies recurring themes
  - Schema integration: new knowledge merged with existing wiki
  - Synaptic pruning: outdated articles marked stale
  - REM dreaming: creative recombination of diverse episodes

Usage::

    consolidator = MemoryConsolidator(
        wiki_dir="/home/claude/agi-hpc/wiki",
        llm_url="http://localhost:8080",
    )
    results = await consolidator.run_cycle()
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConsolidatorConfig:
    """Configuration for the memory consolidator."""

    wiki_dir: str = "/home/claude/agi-hpc/wiki"
    llm_url: str = "http://localhost:8080"
    llm_timeout: int = 120
    db_dsn: str = "dbname=atlas user=claude"
    max_episodes_per_cycle: int = 50
    min_episodes_for_article: int = 3
    similarity_threshold: float = 0.75
    dream_creativity_temp: float = 0.9
    consolidation_temp: float = 0.3
    max_article_length: int = 3000
    stale_days: int = 30


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """An episodic memory (conversation or interaction)."""

    id: str
    timestamp: datetime
    task_description: str
    messages: List[Dict[str, str]]
    success: Optional[bool] = None
    insights: List[str] = field(default_factory=list)
    consolidated: bool = False

    @property
    def text(self) -> str:
        """Flatten messages to plain text for LLM consumption."""
        parts = [f"Task: {self.task_description}"]
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content[:500]}")
        return "\n".join(parts)


@dataclass
class TopicCluster:
    """A cluster of episodes about the same topic."""

    topic: str
    episodes: List[Episode]
    keywords: List[str] = field(default_factory=list)


@dataclass
class ExtractedFact:
    """A fact extracted from episodic memories with certainty scoring."""

    text: str
    certainty: float  # 0-1: how sure this is CORRECT
    confidence: float  # 0-1: how sure this is COMPLETE
    source_episode_ids: List[str] = field(default_factory=list)
    contradicts: Optional[str] = None  # slug of contradicted wiki article, if any
    needs_validation: bool = False

    @property
    def tier(self) -> str:
        """Classify fact into a display tier for the wiki article."""
        if self.certainty >= 0.8 and self.confidence >= 0.7:
            return "established"  # stated as known fact
        if self.certainty >= 0.5:
            return "probable"  # stated with hedging language
        return "unverified"  # flagged for human review


@dataclass
class CertaintyMetrics:
    """Aggregate certainty metrics for a consolidated article."""

    mean_certainty: float  # average certainty across all facts
    mean_confidence: float  # average confidence across all facts
    min_certainty: float  # weakest fact
    source_agreement: float  # fraction of facts confirmed by 2+ episodes
    contradiction_count: int  # facts that conflict with existing wiki
    facts_total: int
    facts_established: int  # certainty >= 0.8
    facts_probable: int  # 0.5 <= certainty < 0.8
    facts_unverified: int  # certainty < 0.5

    @property
    def article_grade(self) -> str:
        """Letter grade for overall article reliability."""
        score = (
            self.mean_certainty * 0.6
            + self.source_agreement * 0.3
            + self.mean_confidence * 0.1
        )
        if score >= 0.8:
            return "A"
        if score >= 0.6:
            return "B"
        if score >= 0.4:
            return "C"
        return "D"


@dataclass
class WikiArticle:
    """A generated or updated wiki article."""

    slug: str
    title: str
    content: str
    source_episodes: List[str]
    confidence: float
    certainty_metrics: Optional[CertaintyMetrics] = None
    is_update: bool = False
    previous_hash: Optional[str] = None


@dataclass
class DreamInsight:
    """A novel insight from creative recombination."""

    text: str
    source_episodes: List[str]
    novelty_score: float
    coherence_score: float


@dataclass
class ConsolidationResult:
    """Summary of a dream cycle."""

    timestamp: datetime
    episodes_processed: int
    clusters_found: int
    articles_created: int
    articles_updated: int
    articles_staled: int
    dream_insights: int
    duration_seconds: float
    mean_certainty: float = 0.0
    mean_confidence: float = 0.0
    plasticity_examples: int = 0
    plasticity_loss: float = 0.0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

ASSESS_CERTAINTY_PROMPT = """\
You are evaluating facts extracted from recent conversations for certainty and confidence.

For each fact, assess:
- **certainty** (0.0-1.0): How likely is this fact CORRECT?
  1.0 = verified by multiple sources or user-confirmed
  0.8 = stated clearly with supporting evidence
  0.5 = plausible but only mentioned once, no confirmation
  0.3 = speculative, hedged language in source conversation
  0.1 = contradicted by other evidence

- **confidence** (0.0-1.0): How COMPLETE is this fact?
  1.0 = fully detailed with numbers, citations, specifics
  0.7 = good detail but missing some context
  0.5 = correct but vague or high-level
  0.3 = partial information, significant gaps
  0.1 = fragment, unclear meaning without context

Scoring criteria:
- Fact mentioned in multiple episodes → certainty +0.2
- User explicitly confirmed or corrected → certainty +0.3 / set to corrected value
- Specific numbers, code, or citations → confidence +0.2
- Hedging language ("maybe", "possibly", "I think") → certainty -0.2
- Contradicts existing knowledge → certainty -0.3, flag contradiction

Facts to assess:
{facts}

Existing wiki knowledge that may be relevant:
{existing_knowledge}

Return a JSON array. Each element:
{{"text": "the fact", "certainty": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "why these scores", "contradicts": null or "slug-of-contradicted-article"}}

Return ONLY valid JSON."""


EXTRACT_TOPICS_PROMPT = """\
You are analyzing a batch of recent conversations to identify distinct topics.
Group them by subject matter and identify the key topic for each group.

Conversations:
{episodes}

Return a JSON array of topic clusters. Each cluster has:
- "topic": a concise topic name (2-5 words, lowercase, suitable as a wiki article slug)
- "episode_ids": array of episode IDs that belong to this cluster
- "keywords": array of 3-5 keywords for this topic

Return ONLY valid JSON, no commentary."""


WRITE_ARTICLE_PROMPT = """\
You are Atlas AI's memory consolidation system. You are writing a wiki article
to capture knowledge learned from recent conversations.

Topic: {topic}

Source conversations:
{episodes}

Existing wiki article (if updating):
{existing}

Write a structured wiki article in markdown following this exact format:

# {title}

[Write a one paragraph summary of what this article covers and why it matters]

## Key Concepts

- **Concept 1**: Definition and explanation
- **Concept 2**: Definition and explanation

## Details

[Write a detailed explanation with specific examples drawn from the conversations.
Include concrete facts, numbers, code snippets, or procedures that were
discussed. This is consolidated knowledge — be precise and cite specifics.]

## Learned Procedures

[If any approaches, methods, or procedures were discussed that worked well,
document them here as step-by-step instructions.]

## Open Questions

[List any unresolved questions or gaps identified during the conversations.]

## Provenance

- Source episodes: {episode_ids}
- Consolidated: {date}
- Confidence: {confidence}

## See Also

- [[related-article-1]]
- [[related-article-2]]

---

Write the COMPLETE article now. Be specific — include actual facts, code,
and details from the conversations. Do not be vague or generic."""


DREAM_PROMPT = """\
You are Atlas AI in REM sleep. Your subconscious has surfaced these fragments
from completely different conversations and domains. They are unrelated — that
is the point.

Fragment 1 ({topic1}):
{fragment1}

Fragment 2 ({topic2}):
{fragment2}

Fragment 3 ({topic3}):
{fragment3}

Find unexpected connections. What analogies bridge these domains? What design
patterns appear in multiple contexts? What novel ideas emerge from combining
these seemingly unrelated topics?

If you find a genuine insight — a connection that would be useful if someone
asked about any of these topics — write it as a concise paragraph. If there
is no meaningful connection, say "No insight" and nothing else."""


FIND_RELATED_PROMPT = """\
Given this list of existing wiki articles and a new article topic,
identify which existing articles should be linked as "See Also".

Existing articles:
{existing_articles}

New article topic: {topic}

Return a JSON array of slug strings for related articles. Return ONLY valid JSON."""


async def _llm_generate(
    url: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    """Call the LLM (llama.cpp /v1/chat/completions endpoint)."""
    payload = {
        "model": "atlas",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM returned {resp.status}: {text[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


async def _fetch_unconsolidated_episodes(dsn: str, limit: int = 50) -> List[Episode]:
    """Fetch recent episodes not yet consolidated."""
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg not installed — using psycopg2 fallback")
        return _fetch_episodes_sync(dsn, limit)

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id, timestamp, user_message, atlas_response,
                   hemisphere, quality_score, metadata
            FROM episodes
            WHERE metadata->>'consolidated' IS NULL
               OR metadata->>'consolidated' = 'false'
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit,
        )
        episodes = []
        for row in rows:
            messages = []
            if row["user_message"]:
                messages.append({"role": "user", "content": row["user_message"]})
            if row["atlas_response"]:
                messages.append({"role": "assistant", "content": row["atlas_response"]})

            meta = row["metadata"] or {}
            insights = meta.get("insights", [])
            if isinstance(insights, str):
                insights = [insights]

            episodes.append(
                Episode(
                    id=str(row["id"]),
                    timestamp=row["timestamp"],
                    task_description=row["user_message"] or "",
                    messages=messages,
                    success=row["quality_score"] is not None
                    and row["quality_score"] > 0.5,
                    insights=insights,
                )
            )
        return episodes
    finally:
        await conn.close()


def _fetch_episodes_sync(dsn: str, limit: int = 50) -> List[Episode]:
    """Synchronous fallback using psycopg2."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, timestamp, user_message, atlas_response,
                       hemisphere, quality_score, metadata
                FROM episodes
                WHERE metadata->>'consolidated' IS NULL
                   OR metadata->>'consolidated' = 'false'
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            episodes = []
            for row in rows:
                messages = []
                if row.get("user_message"):
                    messages.append({"role": "user", "content": row["user_message"]})
                if row.get("atlas_response"):
                    messages.append(
                        {"role": "assistant", "content": row["atlas_response"]}
                    )

                meta = row.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                insights = meta.get("insights", [])

                episodes.append(
                    Episode(
                        id=str(row["id"]),
                        timestamp=row.get("timestamp"),
                        task_description=row.get("user_message") or "",
                        messages=messages,
                        success=row.get("quality_score") is not None
                        and float(row.get("quality_score", 0)) > 0.5,
                        insights=insights if isinstance(insights, list) else [],
                    )
                )
            return episodes
    finally:
        conn.close()


async def _mark_consolidated(dsn: str, episode_ids: List[str]) -> None:
    """Mark episodes as consolidated in the database."""
    if not episode_ids:
        return
    try:
        import asyncpg

        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                UPDATE episodes
                SET metadata = metadata || '{"consolidated": true}'::jsonb
                WHERE id = ANY($1::uuid[])
                """,
                episode_ids,
            )
        finally:
            await conn.close()
    except ImportError:
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE episodes SET metadata = metadata || '{\"consolidated\": true}'::jsonb WHERE id IN %s",
                    (tuple(episode_ids),),
                )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Wiki I/O
# ---------------------------------------------------------------------------


def _article_hash(content: str) -> str:
    """SHA-256 hash of article content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _slug(topic: str, prefix: str = "dream") -> str:
    """Convert a topic name to a wiki slug with dream- prefix.

    Dream-consolidated articles are prefixed with 'dream-' so the RAG
    server identifies them as Tier 0 (highest priority). These represent
    validated knowledge consolidated from actual conversations, scored
    1.5x vs 1.0x for hand-written wiki articles.
    """
    s = topic.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    s = s or "untitled"
    if prefix and not s.startswith(f"{prefix}-"):
        s = f"{prefix}-{s}"
    return s


def _read_existing_article(wiki_dir: Path, slug: str) -> Optional[str]:
    """Read an existing wiki article, or None if it doesn't exist."""
    path = wiki_dir / f"{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_article(wiki_dir: Path, slug: str, content: str) -> Path:
    """Write a wiki article to disk."""
    path = wiki_dir / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _list_existing_slugs(wiki_dir: Path) -> List[str]:
    """List all existing wiki article slugs."""
    return [p.stem for p in wiki_dir.glob("*.md") if p.stem != "index"]


def _update_wiki_index(wiki_dir: Path) -> None:
    """Regenerate the wiki index.md with all articles."""
    slugs = sorted(_list_existing_slugs(wiki_dir))
    lines = ["# AGI-HPC Wiki\n", f"\n*{len(slugs)} articles*\n"]
    for slug in slugs:
        path = wiki_dir / f"{slug}.md"
        # Read first heading for title
        title = slug
        try:
            first_line = path.read_text(encoding="utf-8").split("\n")[0]
            if first_line.startswith("# "):
                title = first_line[2:].strip()
        except Exception:
            pass
        lines.append(f"- [+] [[{slug}]] -- {title}")
    lines.append("")
    (wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core consolidation pipeline
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """
    The dreaming engine. Replays episodic memories and synthesizes
    wiki articles as consolidated knowledge.
    """

    def __init__(self, config: Optional[ConsolidatorConfig] = None):
        self.config = config or ConsolidatorConfig()
        self.wiki_dir = Path(self.config.wiki_dir)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

    async def run_cycle(self) -> ConsolidationResult:
        """Execute one complete dream cycle."""
        start = time.monotonic()
        result = ConsolidationResult(
            timestamp=datetime.now(timezone.utc),
            episodes_processed=0,
            clusters_found=0,
            articles_created=0,
            articles_updated=0,
            articles_staled=0,
            dream_insights=0,
            duration_seconds=0,
        )

        # Stage 1: Episodic Replay — fetch unconsolidated episodes
        logger.info("[dream] Stage 1: Episodic replay — fetching episodes")
        episodes = await _fetch_unconsolidated_episodes(
            self.config.db_dsn, self.config.max_episodes_per_cycle
        )
        result.episodes_processed = len(episodes)

        if len(episodes) < self.config.min_episodes_for_article:
            logger.info(
                "[dream] Only %d episodes (need %d) — skipping cycle",
                len(episodes),
                self.config.min_episodes_for_article,
            )
            result.duration_seconds = time.monotonic() - start
            return result

        logger.info("[dream] Replaying %d episodes", len(episodes))

        # Stage 2: Topic Clustering — LLM groups episodes by subject
        logger.info("[dream] Stage 2: Topic clustering")
        clusters = await self._cluster_episodes(episodes)
        result.clusters_found = len(clusters)
        logger.info("[dream] Found %d topic clusters", len(clusters))

        # Stage 3: Article Synthesis — write/update wiki articles per cluster
        logger.info("[dream] Stage 3: Article synthesis")
        all_certainties = []
        all_confidences = []
        for cluster in clusters:
            try:
                article = await self._synthesize_article(cluster)
                if article:
                    path = _write_article(self.wiki_dir, article.slug, article.content)
                    if article.is_update:
                        result.articles_updated += 1
                        logger.info("[dream] Updated article: %s", path.name)
                    else:
                        result.articles_created += 1
                        logger.info("[dream] Created article: %s", path.name)
                    # Accumulate certainty metrics
                    if article.certainty_metrics:
                        all_certainties.append(article.certainty_metrics.mean_certainty)
                        all_confidences.append(
                            article.certainty_metrics.mean_confidence
                        )
            except Exception as e:
                err = f"Article synthesis failed for '{cluster.topic}': {e}"
                logger.error("[dream] %s", err)
                result.errors.append(err)

        # Stage 4: Creative Dreaming — recombine diverse episodes
        logger.info("[dream] Stage 4: Creative dreaming")
        if len(episodes) >= 6:
            try:
                insights = await self._dream(episodes)
                result.dream_insights = len(insights)
                for insight in insights:
                    logger.info(
                        "[dream] Insight (novelty=%.2f): %s",
                        insight.novelty_score,
                        insight.text[:100],
                    )
            except Exception as e:
                logger.error("[dream] Dreaming failed: %s", e)
                result.errors.append(f"Dreaming failed: {e}")

        # Stage 5: Housekeeping
        logger.info("[dream] Stage 5: Housekeeping")
        _update_wiki_index(self.wiki_dir)

        # Record aggregate certainty
        if all_certainties:
            result.mean_certainty = sum(all_certainties) / len(all_certainties)
            result.mean_confidence = sum(all_confidences) / len(all_confidences)

        # Mark episodes as consolidated
        all_episode_ids = [ep.id for ep in episodes]
        try:
            await _mark_consolidated(self.config.db_dsn, all_episode_ids)
        except Exception as e:
            logger.error("[dream] Failed to mark consolidated: %s", e)
            result.errors.append(f"Mark consolidated failed: {e}")

        # Stage 5b: Synaptic Plasticity — LoRA fine-tune from wiki
        logger.info("[dream] Stage 5b: Synaptic plasticity")
        try:
            from agi.dreaming.synaptic_plasticity import (
                PlasticityConfig,
                run_plasticity_session,
            )

            plasticity_cfg = PlasticityConfig(wiki_dir=self.wiki_dir)
            plasticity_result = run_plasticity_session(plasticity_cfg)

            if plasticity_result.skipped_reason:
                logger.info(
                    "[dream] Plasticity skipped: %s",
                    plasticity_result.skipped_reason,
                )
            else:
                logger.info(
                    "[dream] Plasticity complete: %d examples, " "loss=%.4f, %.0fs",
                    plasticity_result.examples_trained,
                    plasticity_result.training_loss,
                    plasticity_result.duration_seconds,
                )
            result.plasticity_examples = plasticity_result.examples_trained
            result.plasticity_loss = plasticity_result.training_loss
        except ImportError:
            logger.info("[dream] Synaptic plasticity not available (missing deps)")
        except Exception as e:
            logger.warning("[dream] Plasticity failed: %s", e)
            result.errors.append(f"Plasticity failed: {e}")

        result.duration_seconds = time.monotonic() - start
        logger.info(
            "[dream] Cycle complete: %d episodes -> %d created, %d updated, "
            "%d insights in %.1fs",
            result.episodes_processed,
            result.articles_created,
            result.articles_updated,
            result.dream_insights,
            result.duration_seconds,
        )
        return result

    # ----- Stage 2: Clustering -----

    async def _cluster_episodes(self, episodes: List[Episode]) -> List[TopicCluster]:
        """Use LLM to cluster episodes by topic."""
        # Format episodes for the prompt
        ep_text = "\n\n".join(
            f"Episode {ep.id} ({ep.timestamp.isoformat()}):\n{ep.text[:400]}"
            for ep in episodes
        )

        raw = await _llm_generate(
            self.config.llm_url,
            EXTRACT_TOPICS_PROMPT.format(episodes=ep_text),
            temperature=self.config.consolidation_temp,
            max_tokens=2048,
        )

        # Parse JSON response
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```json\s*", "", raw)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            clusters_raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[dream] Failed to parse cluster JSON, falling back")
            # Fallback: treat all episodes as one cluster
            return [
                TopicCluster(
                    topic="general-knowledge",
                    episodes=episodes,
                    keywords=["general"],
                )
            ]

        # Build cluster objects
        ep_map = {ep.id: ep for ep in episodes}
        clusters = []
        for c in clusters_raw:
            cluster_episodes = [
                ep_map[eid] for eid in c.get("episode_ids", []) if eid in ep_map
            ]
            if cluster_episodes:
                clusters.append(
                    TopicCluster(
                        topic=c.get("topic", "unknown"),
                        episodes=cluster_episodes,
                        keywords=c.get("keywords", []),
                    )
                )

        return clusters

    # ----- Stage 3: Article Synthesis with Certainty Assessment -----

    async def _assess_certainty(
        self, cluster: TopicCluster
    ) -> Tuple[List[ExtractedFact], CertaintyMetrics]:
        """Extract facts from episodes and score each for certainty + confidence."""

        # First pass: have LLM extract discrete facts from the episodes
        ep_text = "\n\n".join(
            f"Episode {ep.id}:\n{ep.text[:400]}" for ep in cluster.episodes
        )

        extract_prompt = (
            f"Extract all discrete factual claims from these conversations "
            f"about '{cluster.topic}'. List each fact on its own line, "
            f"prefixed with the episode ID(s) that support it.\n\n"
            f"{ep_text}\n\n"
            f'Return a JSON array: [{{"text": "...", "episode_ids": ["..."]}}]'
        )

        raw_facts = await _llm_generate(
            self.config.llm_url,
            extract_prompt,
            temperature=0.2,
            max_tokens=2048,
        )

        # Parse extracted facts
        try:
            cleaned = re.sub(r"```json\s*", "", raw_facts)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            facts_raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[dream] Could not parse extracted facts, using fallback")
            facts_raw = [{"text": cluster.topic, "episode_ids": []}]

        # Gather existing wiki knowledge for contradiction checking
        existing_slugs = _list_existing_slugs(self.wiki_dir)
        existing_knowledge = ""
        for slug in existing_slugs[:10]:
            article = _read_existing_article(self.wiki_dir, slug)
            if article:
                # Just the first 200 chars of each for context
                existing_knowledge += f"[{slug}]: {article[:200]}\n"

        # Second pass: score each fact for certainty and confidence
        facts_text = json.dumps(facts_raw, indent=2)
        raw_scores = await _llm_generate(
            self.config.llm_url,
            ASSESS_CERTAINTY_PROMPT.format(
                facts=facts_text,
                existing_knowledge=existing_knowledge or "(no existing wiki)",
            ),
            temperature=0.1,
            max_tokens=2048,
        )

        # Parse scored facts
        try:
            cleaned = re.sub(r"```json\s*", "", raw_scores)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            scored = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("[dream] Could not parse certainty scores, using defaults")
            scored = [
                {
                    "text": f["text"],
                    "certainty": 0.5,
                    "confidence": 0.5,
                    "reasoning": "default",
                    "contradicts": None,
                }
                for f in facts_raw
            ]

        # Build ExtractedFact objects
        facts = []
        for item in scored:
            cert = max(0.0, min(1.0, float(item.get("certainty", 0.5))))
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            fact = ExtractedFact(
                text=item.get("text", ""),
                certainty=cert,
                confidence=conf,
                source_episode_ids=item.get("episode_ids", []),
                contradicts=item.get("contradicts"),
                needs_validation=cert < 0.5,
            )
            facts.append(fact)

        # Compute aggregate metrics
        if facts:
            certs = [f.certainty for f in facts]
            confs = [f.confidence for f in facts]
            multi_source = sum(1 for f in facts if len(f.source_episode_ids) >= 2)
            metrics = CertaintyMetrics(
                mean_certainty=sum(certs) / len(certs),
                mean_confidence=sum(confs) / len(confs),
                min_certainty=min(certs),
                source_agreement=multi_source / len(facts) if facts else 0,
                contradiction_count=sum(1 for f in facts if f.contradicts),
                facts_total=len(facts),
                facts_established=sum(1 for f in facts if f.tier == "established"),
                facts_probable=sum(1 for f in facts if f.tier == "probable"),
                facts_unverified=sum(1 for f in facts if f.tier == "unverified"),
            )
        else:
            metrics = CertaintyMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        logger.info(
            "[dream] Certainty assessment for '%s': %d facts "
            "(established=%d, probable=%d, unverified=%d), "
            "mean_cert=%.2f, mean_conf=%.2f, grade=%s",
            cluster.topic,
            metrics.facts_total,
            metrics.facts_established,
            metrics.facts_probable,
            metrics.facts_unverified,
            metrics.mean_certainty,
            metrics.mean_confidence,
            metrics.article_grade,
        )

        return facts, metrics

    async def _synthesize_article(self, cluster: TopicCluster) -> Optional[WikiArticle]:
        """Synthesize a wiki article with certainty-scored facts."""
        slug = _slug(cluster.topic)
        existing = _read_existing_article(self.wiki_dir, slug)
        is_update = existing is not None

        # Assess certainty of extracted facts
        facts, metrics = await self._assess_certainty(cluster)

        # Format episode content for the article writer
        ep_text = "\n\n---\n\n".join(
            f"Episode {ep.id} ({ep.timestamp.isoformat()}):\n{ep.text[:600]}"
            for ep in cluster.episodes
        )

        # Build certainty-aware instructions for the article writer
        fact_instructions = "\n\nFACTS WITH CERTAINTY SCORES:\n"
        for f in facts:
            marker = {
                "established": "[ESTABLISHED]",
                "probable": "[PROBABLE]",
                "unverified": "[NEEDS VALIDATION]",
            }[f.tier]
            fact_instructions += (
                f"  {marker} (cert={f.certainty:.1f}, conf={f.confidence:.1f}) "
                f"{f.text}\n"
            )
            if f.contradicts:
                fact_instructions += (
                    f"    ^ CONTRADICTS existing article: {f.contradicts}\n"
                )

        fact_instructions += (
            "\nIMPORTANT: In the article, present facts according to their tier:\n"
            "- ESTABLISHED facts: state directly as known knowledge\n"
            "- PROBABLE facts: use hedging language ('likely', 'appears to be', 'evidence suggests')\n"
            "- NEEDS VALIDATION facts: place in a separate '## Unverified' section with a note\n"
            "- CONTRADICTIONS: note both the old and new claims and flag for review\n"
        )

        # Find related articles for backlinks
        existing_slugs = _list_existing_slugs(self.wiki_dir)
        related = await self._find_related_articles(cluster.topic, existing_slugs)

        prompt = WRITE_ARTICLE_PROMPT.format(
            topic=cluster.topic,
            title=cluster.topic.replace("-", " ").title(),
            episodes=ep_text + fact_instructions,
            existing=existing or "(no existing article -- creating new)",
            episode_ids=", ".join(ep.id for ep in cluster.episodes),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            confidence=metrics.article_grade,
        )

        content = await _llm_generate(
            self.config.llm_url,
            prompt,
            temperature=self.config.consolidation_temp,
            max_tokens=self.config.max_article_length,
        )

        # Append backlinks if not already present
        if related and "## See Also" not in content:
            see_also = "\n\n## See Also\n\n"
            see_also += "\n".join(f"- [[{s}]]" for s in related)
            content += see_also

        # Add certainty metrics + dreaming metadata footer
        meta = (
            f"\n\n---\n"
            f"\n"
            f"**Episodic Certainty Report**\n"
            f"\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Article grade | **{metrics.article_grade}** |\n"
            f"| Mean certainty | {metrics.mean_certainty:.2f} |\n"
            f"| Mean confidence | {metrics.mean_confidence:.2f} |\n"
            f"| Min certainty (weakest fact) | {metrics.min_certainty:.2f} |\n"
            f"| Source agreement (2+ episodes) | {metrics.source_agreement:.0%} |\n"
            f"| Contradictions | {metrics.contradiction_count} |\n"
            f"| Facts: established / probable / unverified | "
            f"{metrics.facts_established} / {metrics.facts_probable} / {metrics.facts_unverified} |\n"
            f"\n"
            f"*Consolidated by dreaming subsystem on "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
            f"*Source episodes: {len(cluster.episodes)} | "
            f"Keywords: {', '.join(cluster.keywords)}*\n"
        )
        content += meta

        return WikiArticle(
            slug=slug,
            title=cluster.topic,
            content=content,
            source_episodes=[ep.id for ep in cluster.episodes],
            confidence=metrics.mean_certainty,
            certainty_metrics=metrics,
            is_update=is_update,
            previous_hash=_article_hash(existing) if existing else None,
        )

    async def _find_related_articles(
        self, topic: str, existing_slugs: List[str]
    ) -> List[str]:
        """Use LLM to find related existing articles for backlinks."""
        if not existing_slugs:
            return []

        try:
            raw = await _llm_generate(
                self.config.llm_url,
                FIND_RELATED_PROMPT.format(
                    existing_articles="\n".join(f"- {s}" for s in existing_slugs),
                    topic=topic,
                ),
                temperature=0.1,
                max_tokens=256,
            )
            cleaned = re.sub(r"```json\s*", "", raw)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            related = json.loads(cleaned)
            return [s for s in related if s in existing_slugs]
        except Exception:
            return []

    # ----- Stage 4: Creative Dreaming -----

    async def _dream(self, episodes: List[Episode]) -> List[DreamInsight]:
        """Creative recombination of diverse episodic fragments.

        Selects maximally diverse episode triplets using embedding
        distance, generates cross-domain insights, then scores each
        for novelty (vs existing wiki) and coherence (via LLM eval).

        Generates up to 3 insights per cycle from different
        fragment combinations.
        """
        if len(episodes) < 6:
            return []

        # Select diverse triplets using embedding-based diversity
        triplets = self._select_diverse_triplets(episodes, n=3)
        insights: List[DreamInsight] = []

        for picks in triplets:
            # Generate insight via creative LLM prompt
            raw = await _llm_generate(
                self.config.llm_url,
                DREAM_PROMPT.format(
                    topic1=picks[0].task_description[:50],
                    fragment1=picks[0].text[:400],
                    topic2=picks[1].task_description[:50],
                    fragment2=picks[1].text[:400],
                    topic3=picks[2].task_description[:50],
                    fragment3=picks[2].text[:400],
                ),
                temperature=self.config.dream_creativity_temp,
                max_tokens=512,
            )

            if "no insight" in raw.lower():
                continue

            insight_text = raw.strip()

            # Score novelty: embed insight, compare to existing wiki
            novelty = await self._score_novelty(insight_text)

            # Score coherence: ask LLM to evaluate
            coherence = await self._score_coherence(insight_text, picks)

            insight = DreamInsight(
                text=insight_text,
                source_episodes=[p.id for p in picks],
                novelty_score=novelty,
                coherence_score=coherence,
            )

            # Only keep insights above quality threshold
            if novelty > 0.3 and coherence > 0.4:
                insights.append(insight)
                # Save high-quality insights as wiki articles
                if novelty > 0.6 and coherence > 0.6:
                    self._save_dream_insight(insight)

        return insights

    def _select_diverse_triplets(
        self,
        episodes: List[Episode],
        n: int = 3,
    ) -> List[List[Episode]]:
        """Select n triplets of maximally diverse episodes.

        Uses task_description text similarity as a proxy for
        topic diversity. Falls back to random thirds if
        embedding is unavailable.
        """
        import random

        if len(episodes) < 6:
            return []

        try:
            # Group episodes by rough topic using simple
            # keyword hashing for speed
            from collections import defaultdict

            buckets: dict = defaultdict(list)
            for ep in episodes:
                # Hash first 3 significant words as bucket key
                words = [
                    w.lower() for w in ep.task_description.split()[:10] if len(w) > 3
                ]
                key = words[0] if words else "misc"
                buckets[key].append(ep)

            # Pick from different buckets for diversity
            bucket_list = list(buckets.values())
            triplets = []
            for _ in range(n):
                if len(bucket_list) >= 3:
                    chosen_buckets = random.sample(bucket_list, 3)
                    triplet = [random.choice(b) for b in chosen_buckets]
                else:
                    # Fall back to random selection
                    triplet = random.sample(episodes, min(3, len(episodes)))
                triplets.append(triplet)
            return triplets
        except Exception:
            # Fallback: simple thirds
            third = len(episodes) // 3
            return [
                [
                    random.choice(episodes[:third]),
                    random.choice(episodes[third : 2 * third]),
                    random.choice(episodes[2 * third :]),
                ]
                for _ in range(min(n, 2))
            ]

    async def _score_novelty(self, insight_text: str) -> float:
        """Score how novel an insight is vs existing wiki articles.

        Embeds the insight and compares to existing wiki content.
        High similarity to existing articles = low novelty.

        Returns:
            Novelty score 0.0-1.0 (higher = more novel).
        """
        try:
            from pathlib import Path

            wiki_path = Path(self.wiki_dir)
            if not wiki_path.exists():
                return 0.7  # No wiki yet, assume moderate novelty

            articles = list(wiki_path.glob("dream-*.md"))
            if not articles:
                return 0.8  # First insights are novel

            # Compare insight text against existing article content
            # using simple word overlap (fast, no embedding needed)
            insight_words = set(w.lower() for w in insight_text.split() if len(w) > 4)
            if not insight_words:
                return 0.5

            max_overlap = 0.0
            for article_path in articles[:20]:  # Cap at 20
                try:
                    content = article_path.read_text(encoding="utf-8")
                    article_words = set(
                        w.lower() for w in content.split() if len(w) > 4
                    )
                    if article_words:
                        overlap = len(insight_words & article_words) / len(
                            insight_words
                        )
                        max_overlap = max(max_overlap, overlap)
                except Exception:
                    continue

            # High overlap = low novelty
            return max(0.0, min(1.0, 1.0 - max_overlap))
        except Exception:
            return 0.5

    async def _score_coherence(
        self,
        insight_text: str,
        source_episodes: List[Episode],
    ) -> float:
        """Score how coherent and useful an insight is.

        Uses a second LLM pass to evaluate whether the insight
        is genuinely useful or just word salad.

        Returns:
            Coherence score 0.0-1.0 (higher = more coherent).
        """
        eval_prompt = (
            "Rate this creative insight on a scale of 1-10 "
            "for coherence and usefulness. A coherent insight "
            "makes a clear, non-obvious connection between "
            "topics. An incoherent one is vague or forced.\n\n"
            f"Insight: {insight_text[:500]}\n\n"
            "Source topics: "
            + ", ".join(ep.task_description[:30] for ep in source_episodes)
            + "\n\n"
            "Respond with ONLY a number 1-10."
        )

        try:
            raw = await _llm_generate(
                self.config.llm_url,
                eval_prompt,
                temperature=0.1,
                max_tokens=10,
            )
            # Parse the number
            import re

            nums = re.findall(r"\d+", raw.strip())
            if nums:
                score = int(nums[0])
                return min(1.0, max(0.0, score / 10.0))
        except Exception:
            pass
        return 0.5

    def _save_dream_insight(self, insight: DreamInsight) -> None:
        """Save a high-quality dream insight as a wiki article."""
        try:
            from pathlib import Path

            wiki_path = Path(self.wiki_dir)
            wiki_path.mkdir(parents=True, exist_ok=True)

            # Generate slug from first few words
            words = [
                w.lower()
                for w in insight.text.split()[:6]
                if w.isalpha() and len(w) > 2
            ]
            slug = "-".join(words[:4]) or "insight"
            slug = f"dream-insight-{slug}"

            content = (
                f"# Dream Insight: {' '.join(words[:6]).title()}\n\n"
                f"## Insight\n\n{insight.text}\n\n"
                f"## Metrics\n\n"
                f"- Novelty: {insight.novelty_score:.2f}\n"
                f"- Coherence: {insight.coherence_score:.2f}\n\n"
                f"## Source Episodes\n\n"
                f"- {', '.join(insight.source_episodes)}\n\n"
                f"## Provenance\n\n"
                f"Generated during REM-like creative dreaming.\n"
                f"Cross-domain recombination of "
                f"{len(insight.source_episodes)} fragments.\n"
            )

            path = wiki_path / f"{slug}.md"
            path.write_text(content, encoding="utf-8")
            logger.info(
                "[dream] Saved insight: %s (novelty=%.2f)",
                slug,
                insight.novelty_score,
            )
        except Exception:
            logger.warning("[dream] Failed to save insight")
