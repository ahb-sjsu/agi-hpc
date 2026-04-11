# AGI-HPC Knowledge Extractor
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""LLM-based knowledge extraction from documents.

Sends document text to a local LLM and parses structured output:
entities, relationships, key concepts, and summary.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeExtractionConfig:
    """Configuration for the knowledge extractor."""

    llm_url: str = "http://localhost:8082"
    model_name: str = "Qwen3-32B"
    chunk_size: int = 2000
    overlap: int = 200
    timeout: int = 120


@dataclass
class Entity:
    """A named entity extracted from a document."""

    name: str
    entity_type: str  # person, concept, tool, method, organization
    description: str
    source_doc: str = ""


@dataclass
class Relationship:
    """A directed relationship between two entities."""

    subject: str
    predicate: str  # uses, extends, contradicts, implements, depends_on
    object: str
    confidence: float = 1.0
    source_doc: str = ""


@dataclass
class ExtractedKnowledge:
    """Structured knowledge extracted from a document."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)
    summary: str = ""
    source_path: str = ""


EXTRACTION_PROMPT = """\
Analyze the following text and extract structured knowledge.
Return a JSON object with these fields:

{{
  "entities": [
    {{"name": "...", "type": "concept", "description": "..."}}
  ],
  "relationships": [
    {{"subject": "...", "predicate": "uses", "object": "..."}}
  ],
  "key_concepts": ["concept1", "concept2"],
  "summary": "One paragraph summary of the text."
}}

Rules:
- Extract only entities and relationships explicitly stated or strongly implied.
- Use short, specific entity names.
- Predicates should be simple verbs.
- Return valid JSON only, no markdown fences.

Text:
---
{text}
---
"""


class KnowledgeExtractor:
    """Extract structured knowledge from text using a local LLM."""

    def __init__(self, config: Optional[KnowledgeExtractionConfig] = None) -> None:
        self._config = config or KnowledgeExtractionConfig()
        self._url = self._config.llm_url.rstrip("/")

    def extract_from_text(self, text: str, source: str = "") -> ExtractedKnowledge:
        """Extract entities, relationships, and concepts from text."""
        if not text.strip():
            return ExtractedKnowledge(source_path=source)

        chunks = self._chunk_text(text)
        all_knowledge = ExtractedKnowledge(source_path=source)

        for chunk in chunks:
            prompt = EXTRACTION_PROMPT.format(text=chunk[: self._config.chunk_size])
            response = self._call_llm(prompt)
            parsed = self._parse_response(response, source)
            all_knowledge = self._merge(all_knowledge, parsed)

        logger.info(
            "Extracted %d entities, %d relationships from %s",
            len(all_knowledge.entities),
            len(all_knowledge.relationships),
            source or "text",
        )
        return all_knowledge

    def extract_from_file(self, path: str) -> ExtractedKnowledge:
        """Read a file and extract knowledge from its contents."""
        p = Path(path)
        if not p.exists():
            logger.warning("File not found: %s", path)
            return ExtractedKnowledge(source_path=path)

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(p)
        elif suffix in (".md", ".txt", ".rst", ".py", ".go", ".js", ".yaml", ".json"):
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            logger.warning("Unsupported file type: %s", suffix)
            return ExtractedKnowledge(source_path=path)

        return self.extract_from_text(text, source=str(p))

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        size = self._config.chunk_size
        overlap = self._config.overlap
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to local LLM via OpenAI-compatible API."""
        try:
            import requests
        except ImportError:
            logger.error("requests library required for LLM calls")
            return ""

        try:
            resp = requests.post(
                f"{self._url}/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a knowledge extraction assistant. "
                                "Return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "stream": False,
                },
                timeout=self._config.timeout,
            )
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

    def _parse_response(self, response: str, source: str) -> ExtractedKnowledge:
        """Parse LLM JSON response into ExtractedKnowledge."""
        if not response:
            return ExtractedKnowledge(source_path=source)

        text = response.strip()
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

        # Try direct parse
        data = self._try_json(text)
        if not data:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = self._try_json(match.group())
        if not data:
            logger.warning("Could not parse LLM response as JSON")
            return ExtractedKnowledge(summary=text[:200], source_path=source)

        entities = [
            Entity(
                name=e.get("name", ""),
                entity_type=e.get("type", "concept"),
                description=e.get("description", ""),
                source_doc=source,
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]

        relationships = [
            Relationship(
                subject=r.get("subject", ""),
                predicate=r.get("predicate", "related_to"),
                object=r.get("object", ""),
                source_doc=source,
            )
            for r in data.get("relationships", [])
            if r.get("subject") and r.get("object")
        ]

        return ExtractedKnowledge(
            entities=entities,
            relationships=relationships,
            key_concepts=data.get("key_concepts", []),
            summary=data.get("summary", ""),
            source_path=source,
        )

    def _merge(
        self, a: ExtractedKnowledge, b: ExtractedKnowledge
    ) -> ExtractedKnowledge:
        """Merge two extraction results, deduplicating entities by name."""
        seen_entities = {e.name for e in a.entities}
        merged_entities = list(a.entities)
        for e in b.entities:
            if e.name not in seen_entities:
                merged_entities.append(e)
                seen_entities.add(e.name)

        seen_rels = {(r.subject, r.predicate, r.object) for r in a.relationships}
        merged_rels = list(a.relationships)
        for r in b.relationships:
            key = (r.subject, r.predicate, r.object)
            if key not in seen_rels:
                merged_rels.append(r)
                seen_rels.add(key)

        seen_concepts = set(a.key_concepts)
        merged_concepts = list(a.key_concepts)
        for c in b.key_concepts:
            if c not in seen_concepts:
                merged_concepts.append(c)
                seen_concepts.add(c)

        summary = a.summary or b.summary

        return ExtractedKnowledge(
            entities=merged_entities,
            relationships=merged_rels,
            key_concepts=merged_concepts,
            summary=summary,
            source_path=a.source_path or b.source_path,
        )

    @staticmethod
    def _try_json(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import fitz  # pymupdf

            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            logger.warning("pymupdf not installed, cannot read PDF")
            return ""
