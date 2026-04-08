# AGI-HPC Project
# Tests for hybrid search (RRF, wiki lookup, search cascade).

from __future__ import annotations

from pathlib import Path


from agi.common.hybrid_search import (
    HybridSearcher,
    SearchResult,
    WikiIndex,
    extract_keywords,
    reciprocal_rank_fusion,
    _split_identifier,
)

# ------------------------------------------------------------------ #
# Reciprocal Rank Fusion                                              #
# ------------------------------------------------------------------ #


class TestRRF:
    def test_single_list(self) -> None:
        scores = reciprocal_rank_fusion([["a", "b", "c"]])
        assert list(scores.keys()) == ["a", "b", "c"]
        assert scores["a"] > scores["b"] > scores["c"]

    def test_two_agreeing_lists(self) -> None:
        """Same ranking in both lists should amplify scores."""
        scores = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["a", "b", "c"],
            ]
        )
        assert list(scores.keys())[0] == "a"
        # Score of "a" should be 2 * 1/(60+1) = 2/61
        expected_a = 2 * (1.0 / 61)
        assert abs(scores["a"] - expected_a) < 1e-6

    def test_two_disagreeing_lists(self) -> None:
        """Different rankings should produce merged results."""
        scores = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["c", "b", "a"],
            ]
        )
        # All three docs get very similar scores:
        # "a": rank 0 + rank 2 = 1/61 + 1/63
        # "b": rank 1 + rank 1 = 2/62
        # "c": rank 2 + rank 0 = 1/63 + 1/61
        # "a" and "c" tie; "b" is very close
        assert len(scores) == 3
        vals = list(scores.values())
        assert max(vals) - min(vals) < 0.001  # all very close

    def test_disjoint_lists(self) -> None:
        """Non-overlapping lists should include all docs."""
        scores = reciprocal_rank_fusion(
            [
                ["a", "b"],
                ["c", "d"],
            ]
        )
        assert len(scores) == 4

    def test_empty_list(self) -> None:
        scores = reciprocal_rank_fusion([[], []])
        assert scores == {}

    def test_custom_k(self) -> None:
        scores_k1 = reciprocal_rank_fusion([["a", "b"]], k=1)
        scores_k100 = reciprocal_rank_fusion([["a", "b"]], k=100)
        # Lower k gives higher absolute scores
        assert scores_k1["a"] > scores_k100["a"]

    def test_three_way_fusion(self) -> None:
        scores = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["b", "c", "a"],
                ["c", "a", "b"],
            ]
        )
        # All docs appear at all ranks, but with different distributions
        assert len(scores) == 3
        # All should have similar scores (each appears once at each rank)
        vals = list(scores.values())
        assert max(vals) - min(vals) < 0.001


# ------------------------------------------------------------------ #
# Wiki Index                                                          #
# ------------------------------------------------------------------ #


class TestWikiIndex:
    def test_load_empty_dir(self, tmp_path: Path) -> None:
        wiki = WikiIndex(str(tmp_path))
        count = wiki.load()
        assert count == 0
        assert wiki.article_count == 0

    def test_load_articles(self, tmp_path: Path) -> None:
        (tmp_path / "safety-gateway.md").write_text(
            "# Safety Gateway\n\nThe safety gateway provides three layers..."
        )
        (tmp_path / "event-fabric.md").write_text(
            "# Event Fabric\n\nNATS JetStream backbone for messaging."
        )
        wiki = WikiIndex(str(tmp_path))
        count = wiki.load()
        assert count == 2
        assert wiki.article_count == 2

    def test_lookup_by_keyword(self, tmp_path: Path) -> None:
        (tmp_path / "safety-gateway.md").write_text(
            "# Safety Gateway\n\nThree-layer safety architecture."
        )
        (tmp_path / "kv-cache.md").write_text(
            "# KV Cache Compression\n\nTurboQuant for LLM inference."
        )
        wiki = WikiIndex(str(tmp_path))
        wiki.load()

        results = wiki.lookup("safety gateway")
        assert len(results) >= 1
        assert results[0].slug == "safety-gateway"

    def test_lookup_no_match(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text("# Test\n\nContent.")
        wiki = WikiIndex(str(tmp_path))
        wiki.load()
        results = wiki.lookup("quantum computing blockchain")
        assert len(results) == 0

    def test_backlinks_extracted(self, tmp_path: Path) -> None:
        (tmp_path / "safety.md").write_text(
            "# Safety\n\nRelated to [[event-fabric]] and [[memory]]."
        )
        wiki = WikiIndex(str(tmp_path))
        wiki.load()
        article = wiki._articles["safety"]
        assert "event-fabric" in article.backlinks
        assert "memory" in article.backlinks

    def test_nonexistent_dir(self) -> None:
        wiki = WikiIndex("/nonexistent/path")
        count = wiki.load()
        assert count == 0


# ------------------------------------------------------------------ #
# Search Result                                                       #
# ------------------------------------------------------------------ #


class TestSearchResult:
    def test_to_dict(self) -> None:
        r = SearchResult(
            chunk_id="abc123",
            repo="agi-hpc",
            file="src/safety.py",
            content="Safety code",
            score=0.95,
            source="hybrid",
        )
        d = r.to_dict()
        assert d["repo"] == "agi-hpc"
        assert d["score"] == 0.95
        assert d["source"] == "hybrid"

    def test_provenance_tracking(self) -> None:
        r = SearchResult(
            chunk_id="x",
            repo="r",
            file="f",
            content="c",
            score=0.5,
            source="hybrid",
            vector_rank=2,
            fts_rank=5,
        )
        assert r.vector_rank == 2
        assert r.fts_rank == 5


# ------------------------------------------------------------------ #
# Hybrid Searcher (unit tests, no DB)                                 #
# ------------------------------------------------------------------ #


class TestKeywordExtraction:
    def test_split_camel_case(self) -> None:
        result = _split_identifier("TurboQuantKV")
        assert "turbo" in result
        assert "quant" in result
        assert "kv" in result

    def test_split_snake_case(self) -> None:
        assert _split_identifier("safety_gateway") == ["safety", "gateway"]

    def test_split_mixed(self) -> None:
        result = _split_identifier("PCAMatryoshkaPipeline")
        assert "matryoshka" in result
        assert "pipeline" in result

    def test_extract_from_python(self) -> None:
        code = """
from agi.safety.deme_gateway import SafetyGateway
import numpy as np

class TurboQuantKV:
    def compress(self, tensor):
        pass
"""
        kw = extract_keywords(code, "src/agi/meta/llm/turboquant_kv.py")
        assert "safety" in kw.lower()
        assert "turbo" in kw.lower()
        assert "compress" in kw.lower()
        # File path components
        assert "turboquant" in kw.lower()

    def test_extract_weights_imports(self) -> None:
        code = "from turboquant_pro import PCAMatryoshka"
        kw = extract_keywords(code)
        # imports are double-weighted — "turboquant" appears at least twice
        count = kw.lower().count("turboquant")
        assert count >= 2

    def test_extract_empty(self) -> None:
        assert extract_keywords("") == ""

    def test_extract_plain_text(self) -> None:
        text = "The safety gateway provides three layers of protection."
        kw = extract_keywords(text)
        assert "safety" in kw.lower()
        assert "gateway" in kw.lower()


class TestHybridSearcherUnit:
    def test_stats(self, tmp_path: Path) -> None:
        searcher = HybridSearcher(
            db_dsn="dbname=test",
            wiki_dir=str(tmp_path),
        )
        stats = searcher.stats()
        assert "wiki_articles" in stats
        assert "hamming_ready" in stats
        assert "fts_weight" in stats

    def test_wiki_tier(self, tmp_path: Path) -> None:
        (tmp_path / "safety-gateway.md").write_text(
            "# Safety Gateway\n\nThree layers of protection."
        )
        searcher = HybridSearcher(
            db_dsn="dbname=test",
            wiki_dir=str(tmp_path),
        )
        # Wiki search should work without DB
        results = searcher._search_wiki("safety gateway")
        assert len(results) >= 1
        assert results[0].source == "wiki"
        assert "safety" in results[0].content.lower()

    def test_fts_without_db(self) -> None:
        """FTS should return empty list if no DB connection."""
        searcher = HybridSearcher(
            db_dsn="dbname=nonexistent_test_db_xyz",
            wiki_dir="/nonexistent",
        )
        results = searcher._search_fts("test query", top_k=5)
        assert results == []
