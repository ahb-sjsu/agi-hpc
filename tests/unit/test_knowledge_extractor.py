# AGI-HPC Knowledge Extractor Tests
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agi.memory.knowledge.extractor import (
    Entity,
    ExtractedKnowledge,
    KnowledgeExtractionConfig,
    KnowledgeExtractor,
    Relationship,
)

MOCK_LLM_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "name": "PCA",
                "type": "method",
                "description": "Principal Component Analysis",
            },
            {
                "name": "Matryoshka",
                "type": "concept",
                "description": "Nested representation learning",
            },
            {
                "name": "BGE-M3",
                "type": "tool",
                "description": "Multilingual embedding model",
            },
        ],
        "relationships": [
            {"subject": "PCA", "predicate": "extends", "object": "Matryoshka"},
            {"subject": "BGE-M3", "predicate": "uses", "object": "PCA"},
        ],
        "key_concepts": ["embedding compression", "dimensionality reduction"],
        "summary": "PCA rotation enables Matryoshka-like truncation.",
    }
)


@pytest.fixture
def extractor():
    return KnowledgeExtractor(KnowledgeExtractionConfig(llm_url="http://test:8080"))


class TestParseResponse:
    def test_valid_json(self, extractor):
        result = extractor._parse_response(MOCK_LLM_RESPONSE, "test.md")
        assert len(result.entities) == 3
        assert len(result.relationships) == 2
        assert len(result.key_concepts) == 2
        assert "PCA" in result.summary
        assert result.entities[0].name == "PCA"
        assert result.entities[0].entity_type == "method"

    def test_json_in_markdown_fences(self, extractor):
        wrapped = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        result = extractor._parse_response(wrapped, "test.md")
        assert len(result.entities) == 3

    def test_empty_response(self, extractor):
        result = extractor._parse_response("", "test.md")
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    def test_invalid_json_fallback(self, extractor):
        result = extractor._parse_response("This is not JSON at all.", "test.md")
        assert len(result.entities) == 0
        assert result.summary == "This is not JSON at all."

    def test_partial_json(self, extractor):
        partial = json.dumps(
            {"entities": [{"name": "X", "type": "concept", "description": "test"}]}
        )
        result = extractor._parse_response(partial, "test.md")
        assert len(result.entities) == 1
        assert result.entities[0].name == "X"

    def test_entity_without_name_skipped(self, extractor):
        data = json.dumps({"entities": [{"name": "", "type": "x", "description": "y"}]})
        result = extractor._parse_response(data, "test.md")
        assert len(result.entities) == 0

    def test_relationship_without_subject_skipped(self, extractor):
        data = json.dumps(
            {"relationships": [{"subject": "", "predicate": "x", "object": "y"}]}
        )
        result = extractor._parse_response(data, "test.md")
        assert len(result.relationships) == 0


class TestChunking:
    def test_short_text_single_chunk(self, extractor):
        chunks = extractor._chunk_text("Hello world")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, extractor):
        extractor._config.chunk_size = 100
        extractor._config.overlap = 20
        text = "x" * 250
        chunks = extractor._chunk_text(text)
        assert len(chunks) >= 3


class TestMerge:
    def test_deduplicates_entities(self, extractor):
        a = ExtractedKnowledge(
            entities=[Entity("A", "concept", "first", "doc1")],
            relationships=[],
            key_concepts=["x"],
            summary="Summary A",
        )
        b = ExtractedKnowledge(
            entities=[
                Entity("A", "concept", "duplicate", "doc2"),
                Entity("B", "tool", "new", "doc2"),
            ],
            relationships=[],
            key_concepts=["x", "y"],
            summary="Summary B",
        )
        merged = extractor._merge(a, b)
        assert len(merged.entities) == 2
        assert merged.entities[0].description == "first"  # kept original
        assert len(merged.key_concepts) == 2
        assert merged.summary == "Summary A"

    def test_deduplicates_relationships(self, extractor):
        r1 = Relationship("A", "uses", "B")
        r2 = Relationship("A", "uses", "B")  # duplicate
        r3 = Relationship("A", "extends", "C")
        a = ExtractedKnowledge(relationships=[r1])
        b = ExtractedKnowledge(relationships=[r2, r3])
        merged = extractor._merge(a, b)
        assert len(merged.relationships) == 2


class TestExtractFromText:
    @patch("agi.memory.knowledge.extractor.KnowledgeExtractor._call_llm")
    def test_basic_extraction(self, mock_llm, extractor):
        mock_llm.return_value = MOCK_LLM_RESPONSE
        result = extractor.extract_from_text("Some document text", "doc.md")
        assert len(result.entities) == 3
        assert len(result.relationships) == 2
        assert result.source_path == "doc.md"
        mock_llm.assert_called_once()

    def test_empty_text(self, extractor):
        result = extractor.extract_from_text("", "empty.md")
        assert len(result.entities) == 0


class TestExtractFromFile:
    def test_missing_file(self, extractor):
        result = extractor.extract_from_file("/nonexistent/file.md")
        assert len(result.entities) == 0

    def test_unsupported_extension(self, extractor):
        result = extractor.extract_from_file("/some/file.exe")
        assert len(result.entities) == 0
