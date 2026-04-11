# AGI-HPC Knowledge Graph Tests
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
from __future__ import annotations

import pytest

from agi.memory.knowledge.extractor import (
    Entity,
    ExtractedKnowledge,
    Relationship,
)
from agi.memory.knowledge.graph import KnowledgeGraph, KnowledgeGraphConfig


@pytest.fixture
def graph():
    config = KnowledgeGraphConfig(use_sqlite=True, sqlite_path=":memory:")
    return KnowledgeGraph(config)


@pytest.fixture
def sample_knowledge():
    return ExtractedKnowledge(
        entities=[
            Entity("PCA", "method", "Principal Component Analysis", "doc.md"),
            Entity("BGE-M3", "tool", "Embedding model", "doc.md"),
            Entity("TurboQuant", "tool", "Compression toolkit", "doc.md"),
        ],
        relationships=[
            Relationship("PCA", "used_by", "TurboQuant", 0.9, "doc.md"),
            Relationship("BGE-M3", "produces", "embeddings", 1.0, "doc.md"),
        ],
        key_concepts=["compression", "embeddings"],
        summary="PCA-Matryoshka compression pipeline.",
        source_path="doc.md",
    )


class TestStore:
    def test_store_returns_id(self, graph, sample_knowledge):
        doc_id = graph.store(sample_knowledge)
        assert doc_id
        assert len(doc_id) == 36  # UUID format

    def test_store_persists_entities(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        stats = graph.get_stats()
        assert stats["entities"] == 3
        assert stats["relationships"] == 2
        assert stats["documents"] == 1

    def test_store_empty_knowledge(self, graph):
        doc_id = graph.store(ExtractedKnowledge())
        assert doc_id
        stats = graph.get_stats()
        assert stats["entities"] == 0
        assert stats["documents"] == 1


class TestQueryEntity:
    def test_find_existing(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        entity = graph.query_entity("PCA")
        assert entity is not None
        assert entity.name == "PCA"
        assert entity.entity_type == "method"

    def test_not_found(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        entity = graph.query_entity("NonexistentEntity")
        assert entity is None

    def test_empty_graph(self, graph):
        entity = graph.query_entity("anything")
        assert entity is None


class TestQueryRelationships:
    def test_find_by_subject(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        rels = graph.query_relationships("PCA")
        assert len(rels) == 1
        assert rels[0].predicate == "used_by"
        assert rels[0].object == "TurboQuant"

    def test_find_by_object(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        rels = graph.query_relationships("TurboQuant")
        assert len(rels) == 1
        assert rels[0].subject == "PCA"

    def test_filter_by_predicate(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        rels = graph.query_relationships("PCA", predicate="used_by")
        assert len(rels) == 1
        rels_wrong = graph.query_relationships("PCA", predicate="contradicts")
        assert len(rels_wrong) == 0

    def test_no_matches(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        rels = graph.query_relationships("Unknown")
        assert len(rels) == 0


class TestStats:
    def test_empty_graph(self, graph):
        stats = graph.get_stats()
        assert stats == {"entities": 0, "relationships": 0, "documents": 0}

    def test_after_store(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        stats = graph.get_stats()
        assert stats["entities"] == 3
        assert stats["relationships"] == 2
        assert stats["documents"] == 1

    def test_multiple_stores(self, graph, sample_knowledge):
        graph.store(sample_knowledge)
        graph.store(
            ExtractedKnowledge(
                entities=[Entity("NewEntity", "concept", "test", "doc2.md")],
                relationships=[],
                key_concepts=[],
                summary="second doc",
                source_path="doc2.md",
            )
        )
        stats = graph.get_stats()
        assert stats["entities"] == 4
        assert stats["documents"] == 2
