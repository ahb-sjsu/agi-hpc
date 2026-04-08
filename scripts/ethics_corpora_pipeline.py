#!/usr/bin/env python3
"""
Ethics Corpora Pipeline for Atlas AI RAG System
================================================

Fetches multi-civilizational ethics corpora and indexes them into
PostgreSQL/pgvector on Atlas for RAG retrieval.

Corpora handled:
  1. Sefaria Jewish Library (Hebrew/Aramaic)
  2. Chinese Classics (Classical Chinese)
  3. Islamic Texts (Arabic)
  4. Dear Abby (English)
  5. Perseus Digital Library (Ancient Greek/Latin)
  6. Pali Canon / Tipitaka (Pali)
  7. UN Human Rights Declarations (multilingual)

Usage:
    taskset -c 36-39 /home/claude/env/bin/python /archive/ethics-corpora/pipeline.py 2>&1 | tee /archive/ethics-corpora/pipeline.log

Author: Ethics RAG pipeline for Atlas workstation
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/archive/ethics-corpora")
SQND_DIR = Path("/archive/ahb-sjsu/sqnd-probe")
DB_NAME = "atlas"
DB_USER = "claude"
CHUNK_SIZE_MIN = 200
CHUNK_SIZE_MAX = 1000
CHUNK_SIZE_TARGET = 600
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "pipeline.log", mode="a"),
    ],
)
log = logging.getLogger("ethics_pipeline")


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class EthicsChunk:
    corpus: str
    tradition: str
    language: str
    period: str
    century: Optional[int]
    source_ref: str
    content: str


# ============================================================================
# Database setup
# ============================================================================

def get_db_conn():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER)


def create_table():
    log.info("Creating ethics_chunks table if not exists...")
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ethics_chunks (
            id SERIAL PRIMARY KEY,
            corpus TEXT NOT NULL,
            tradition TEXT NOT NULL,
            language TEXT NOT NULL,
            period TEXT,
            century INT,
            source_ref TEXT,
            content TEXT NOT NULL,
            embedding vector(1024)
        );
    """)
    # Create indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ethics_corpus ON ethics_chunks(corpus);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ethics_tradition ON ethics_chunks(tradition);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ethics_language ON ethics_chunks(language);
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("Table created/verified.")


def count_corpus_rows(corpus_name: str) -> int:
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE corpus = %s", (corpus_name,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def insert_chunks(chunks: list[EthicsChunk], batch_size: int = 1000):
    """Insert chunks without embeddings (embeddings added later)."""
    if not chunks:
        return 0
    conn = get_db_conn()
    cur = conn.cursor()
    inserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        values = [
            (c.corpus, c.tradition, c.language, c.period, c.century, c.source_ref, c.content)
            for c in batch
        ]
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO ethics_chunks (corpus, tradition, language, period, century, source_ref, content)
               VALUES %s""",
            values,
            template="(%s, %s, %s, %s, %s, %s, %s)",
        )
        inserted += len(batch)
        if inserted % 5000 == 0:
            log.info(f"  Inserted {inserted}/{len(chunks)} chunks...")
    conn.commit()
    cur.close()
    conn.close()
    return inserted


# ============================================================================
# Text chunking
# ============================================================================

def chunk_text(text: str, max_size: int = CHUNK_SIZE_MAX, min_size: int = CHUNK_SIZE_MIN) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if not text or len(text.strip()) < min_size:
        if text and len(text.strip()) >= 50:
            return [text.strip()]
        return []

    text = text.strip()
    if len(text) <= max_size:
        return [text]

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?\u3002\u3001\uff01\uff1f])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_size:
            current = (current + " " + sent).strip() if current else sent.strip()
        else:
            if len(current) >= min_size:
                chunks.append(current)
            current = sent.strip()
            # If single sentence is too long, force-split at max_size
            while len(current) > max_size:
                # Find a space near max_size to split
                split_pos = current.rfind(" ", 0, max_size)
                if split_pos < min_size:
                    split_pos = max_size
                chunks.append(current[:split_pos].strip())
                current = current[split_pos:].strip()

    if current and len(current) >= min_size:
        chunks.append(current)
    elif current and chunks:
        # Append short trailing text to last chunk
        chunks[-1] = chunks[-1] + " " + current

    return chunks


# ============================================================================
# Corpus 1: Sefaria Jewish Library
# ============================================================================

SEFARIA_PERIOD_MAP = {
    "Tanakh": ("BIBLICAL", -6),
    "Torah": ("BIBLICAL", -12),
    "Prophets": ("BIBLICAL", -8),
    "Writings": ("BIBLICAL", -4),
    "Mishnah": ("TANNAITIC", 2),
    "Tosefta": ("TANNAITIC", 2),
    "Talmud": ("AMORAIC", 5),
    "Bavli": ("AMORAIC", 5),
    "Yerushalmi": ("AMORAIC", 4),
    "Midrash": ("AMORAIC", 5),
    "Halakhah": ("RISHONIM", 12),
    "Kabbalah": ("RISHONIM", 13),
    "Philosophy": ("RISHONIM", 12),
    "Chasidut": ("ACHRONIM", 18),
    "Musar": ("ACHRONIM", 19),
    "Responsa": ("ACHRONIM", 17),
}

ARAMAIC_CATEGORIES = {"Talmud", "Bavli", "Yerushalmi"}


def fetch_sefaria() -> list[EthicsChunk]:
    """Clone Sefaria-Export and parse JSON files."""
    corpus_dir = BASE_DIR / "sefaria"
    export_dir = corpus_dir / "Sefaria-Export"
    chunks = []

    if not export_dir.exists():
        log.info("Cloning Sefaria-Export (--depth 1)...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Sefaria/Sefaria-Export.git",
             str(export_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error(f"Sefaria clone failed: {result.stderr}")
            return []
        log.info("Sefaria clone complete.")
    else:
        log.info("Sefaria-Export already cloned.")

    json_dir = export_dir / "json"
    if not json_dir.exists():
        log.error(f"Sefaria JSON dir not found: {json_dir}")
        return []

    # Walk JSON directory structure
    file_count = 0
    max_files = 100000  # Safety limit

    for root, dirs, files in os.walk(json_dir):
        root_path = Path(root)
        rel = root_path.relative_to(json_dir)
        parts = rel.parts

        # Determine category from directory structure
        category = parts[0] if parts else "Unknown"
        period_info = SEFARIA_PERIOD_MAP.get(category, ("UNKNOWN", None))
        period, century = period_info

        lang = "aramaic" if category in ARAMAIC_CATEGORIES else "hebrew"

        for fname in files:
            if not fname.endswith(".json"):
                continue
            if file_count >= max_files:
                break

            fpath = root_path / fname
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Extract text from various Sefaria JSON formats
            texts = _extract_sefaria_texts(data, str(rel / fname))
            for ref, text in texts:
                for chunk in chunk_text(text):
                    chunks.append(EthicsChunk(
                        corpus="sefaria",
                        tradition="jewish",
                        language=lang,
                        period=period,
                        century=century,
                        source_ref=ref,
                        content=chunk,
                    ))

            file_count += 1
            if file_count % 1000 == 0:
                log.info(f"  Sefaria: processed {file_count} files, {len(chunks)} chunks so far")

    log.info(f"Sefaria: {len(chunks)} chunks from {file_count} files")
    return chunks


def _extract_sefaria_texts(data: dict, ref_prefix: str) -> list[tuple[str, str]]:
    """Extract text content from Sefaria JSON structure."""
    results = []

    if isinstance(data, dict):
        # Common Sefaria format: {"text": [...], "he": [...]}
        for key in ("text", "he"):
            if key in data:
                _flatten_sefaria_text(data[key], f"{ref_prefix}", results, key)

        # Version format: {"versions": [{"text": [...], "language": "he"}]}
        if "versions" in data and isinstance(data["versions"], list):
            for ver in data["versions"]:
                if isinstance(ver, dict) and "text" in ver:
                    lang_tag = ver.get("language", "he")
                    _flatten_sefaria_text(ver["text"], ref_prefix, results, lang_tag)
    elif isinstance(data, list):
        _flatten_sefaria_text(data, ref_prefix, results, "text")

    return results


def _flatten_sefaria_text(obj, ref: str, results: list, lang_tag: str, depth: int = 0):
    """Recursively flatten nested Sefaria text arrays."""
    if depth > 10:
        return
    if isinstance(obj, str):
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', '', obj).strip()
        if text and len(text) >= 50:
            results.append((f"{ref}:{lang_tag}", text))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _flatten_sefaria_text(item, f"{ref}:{i+1}", results, lang_tag, depth + 1)


# ============================================================================
# Corpus 2: Chinese Classics
# ============================================================================

CHINESE_TEXTS = [
    # (text_id, name, period, century, chapters)
    ("ctp:analects", "Analects", "CONFUCIAN", -5, [
        "xue-er", "wei-zheng", "ba-yi", "li-ren", "gong-ye-chang",
        "yong-ye", "shu-er", "tai-bo", "zi-han", "xiang-dang",
        "xian-jin", "yan-yuan", "zi-lu", "xian-wen", "wei-ling-gong",
        "ji-shi", "yang-huo", "wei-zi", "zi-zhang", "yao-yue",
    ]),
    ("ctp:mengzi", "Mencius", "CONFUCIAN", -4, [
        "liang-hui-wang-i", "liang-hui-wang-ii", "gong-sun-chou-i",
        "gong-sun-chou-ii", "teng-wen-gong-i", "teng-wen-gong-ii",
        "li-lou-i", "li-lou-ii", "wan-zhang-i", "wan-zhang-ii",
        "gao-zi-i", "gao-zi-ii", "jin-xin-i", "jin-xin-ii",
    ]),
    ("ctp:dao-de-jing", "Dao De Jing", "DAOIST", -6, [
        str(i) for i in range(1, 82)
    ]),
    ("ctp:zhuangzi", "Zhuangzi", "DAOIST", -4, [
        "xiao-yao-you", "qi-wu-lun", "yang-sheng-zhu", "ren-jian-shi",
        "de-chong-fu", "da-zong-shi", "ying-di-wang",
    ]),
    ("ctp:xunzi", "Xunzi", "CONFUCIAN", -3, [
        "quan-xue", "xiu-shen", "bu-gou", "rong-ru", "fei-xiang",
    ]),
    ("ctp:mozi", "Mozi", "MOHIST", -5, [
        "shang-xian-shang", "shang-xian-zhong", "shang-xian-xia",
        "shang-tong-shang", "shang-tong-zhong", "shang-tong-xia",
        "jian-ai-shang", "jian-ai-zhong", "jian-ai-xia",
        "fei-gong-shang", "fei-gong-zhong", "fei-gong-xia",
    ]),
    ("ctp:han-feizi", "Han Feizi", "LEGALIST", -3, [
        "chu-jian-qin-nan", "cun-han", "nan-yan", "ai-chen",
    ]),
]


def fetch_chinese_classics() -> list[EthicsChunk]:
    """Fetch Chinese classics from ctext.org API."""
    corpus_dir = BASE_DIR / "chinese_classics"
    cache_file = corpus_dir / "chinese_texts.json"
    chunks = []

    # Check if sqnd-probe has cached data
    sqnd_cache = SQND_DIR / "data" / "raw" / "chinese" / "chinese_native.json"
    if sqnd_cache.exists():
        log.info(f"Using cached Chinese data from sqnd-probe: {sqnd_cache}")
        with open(sqnd_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        for entry in cached:
            text = entry.get("text", "")
            for chunk in chunk_text(text):
                chunks.append(EthicsChunk(
                    corpus="chinese_classics",
                    tradition=entry.get("tradition", "confucian"),
                    language="classical_chinese",
                    period=entry.get("period", "CONFUCIAN"),
                    century=entry.get("century"),
                    source_ref=entry.get("ref", ""),
                    content=chunk,
                ))
        log.info(f"Chinese classics from cache: {len(chunks)} chunks")
        return chunks

    if cache_file.exists():
        log.info(f"Using cached Chinese data: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            all_passages = json.load(f)
    else:
        log.info("Fetching Chinese classics from ctext.org API...")
        all_passages = []
        for text_id, name, period, century, chapters in CHINESE_TEXTS:
            log.info(f"  Fetching {name} ({len(chapters)} chapters)...")
            for ch in chapters:
                url = f"https://api.ctext.org/gettext?urn={text_id}/{ch}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "EthicsCorpora/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            passage = {
                                "text": item.get("text", ""),
                                "tradition": period.lower(),
                                "period": period,
                                "century": century,
                                "ref": f"{name}/{ch}",
                                "source": name,
                            }
                            if passage["text"]:
                                all_passages.append(passage)
                    time.sleep(1.0)  # Rate limit
                except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
                    log.warning(f"    Failed to fetch {name}/{ch}: {e}")
                    time.sleep(2.0)

        # Save cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_passages, f, ensure_ascii=False, indent=2)
        log.info(f"  Cached {len(all_passages)} passages to {cache_file}")

    for entry in all_passages:
        text = entry.get("text", "")
        for chunk in chunk_text(text):
            chunks.append(EthicsChunk(
                corpus="chinese_classics",
                tradition=entry.get("tradition", "confucian"),
                language="classical_chinese",
                period=entry.get("period", "CONFUCIAN"),
                century=entry.get("century"),
                source_ref=entry.get("ref", ""),
                content=chunk,
            ))

    log.info(f"Chinese classics: {len(chunks)} chunks from {len(all_passages)} passages")
    return chunks


# ============================================================================
# Corpus 3: Islamic Texts
# ============================================================================

def fetch_islamic() -> list[EthicsChunk]:
    """Load Islamic texts from sqnd-probe cache or create minimal set."""
    corpus_dir = BASE_DIR / "islamic"
    chunks = []

    # Check sqnd-probe cache
    sqnd_cache = SQND_DIR / "data" / "raw" / "islamic" / "islamic_native.json"
    if sqnd_cache.exists():
        log.info(f"Using cached Islamic data from sqnd-probe: {sqnd_cache}")
        with open(sqnd_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        for entry in cached:
            text = entry.get("text", "")
            for chunk in chunk_text(text):
                chunks.append(EthicsChunk(
                    corpus="islamic",
                    tradition="islamic",
                    language="arabic",
                    period=entry.get("period", "QURANIC"),
                    century=entry.get("century", 7),
                    source_ref=entry.get("ref", ""),
                    content=chunk,
                ))
        log.info(f"Islamic texts from cache: {len(chunks)} chunks")
        return chunks

    # Fetch Quran text from open API (quran.com API v4)
    cache_file = corpus_dir / "quran.json"
    if cache_file.exists():
        log.info(f"Using cached Quran data: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            surahs = json.load(f)
    else:
        log.info("Fetching Quran from api.quran.com...")
        surahs = []
        # Fetch Arabic text of all 114 surahs
        for surah_num in range(1, 115):
            url = f"https://api.quran.com/api/v4/quran/verses/uthmani?chapter_number={surah_num}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "EthicsCorpora/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                verses = data.get("verses", [])
                for v in verses:
                    surahs.append({
                        "text": v.get("text_uthmani", ""),
                        "ref": f"Quran {surah_num}:{v.get('verse_key', '')}",
                    })
                if surah_num % 20 == 0:
                    log.info(f"  Fetched {surah_num}/114 surahs...")
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  Failed surah {surah_num}: {e}")
                time.sleep(2.0)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(surahs, f, ensure_ascii=False, indent=2)
        log.info(f"  Cached {len(surahs)} Quran verses")

    for entry in surahs:
        text = entry.get("text", "")
        if text and len(text.strip()) >= 30:
            # Quran verses are typically short, group several together
            chunks.append(EthicsChunk(
                corpus="islamic",
                tradition="islamic",
                language="arabic",
                period="QURANIC",
                century=7,
                source_ref=entry.get("ref", ""),
                content=text.strip(),
            ))

    # Group short Quran verses into larger chunks for better embedding
    grouped = _group_short_chunks(chunks, max_size=CHUNK_SIZE_MAX)
    log.info(f"Islamic texts: {len(grouped)} chunks")
    return grouped


def _group_short_chunks(chunks: list[EthicsChunk], max_size: int = 800) -> list[EthicsChunk]:
    """Group consecutive short chunks from same corpus into larger ones."""
    if not chunks:
        return []
    grouped = []
    current_text = ""
    current_refs = []
    template = chunks[0]

    for c in chunks:
        if len(current_text) + len(c.content) + 1 <= max_size:
            current_text = (current_text + " " + c.content).strip()
            current_refs.append(c.source_ref)
        else:
            if current_text and len(current_text) >= CHUNK_SIZE_MIN:
                grouped.append(EthicsChunk(
                    corpus=template.corpus,
                    tradition=template.tradition,
                    language=template.language,
                    period=template.period,
                    century=template.century,
                    source_ref="; ".join(current_refs[:3]),
                    content=current_text,
                ))
            current_text = c.content
            current_refs = [c.source_ref]

    if current_text and len(current_text) >= CHUNK_SIZE_MIN:
        grouped.append(EthicsChunk(
            corpus=template.corpus,
            tradition=template.tradition,
            language=template.language,
            period=template.period,
            century=template.century,
            source_ref="; ".join(current_refs[:3]),
            content=current_text,
        ))

    return grouped


# ============================================================================
# Corpus 4: Dear Abby
# ============================================================================

def fetch_dear_abby() -> list[EthicsChunk]:
    """Load Dear Abby advice column from sqnd-probe."""
    chunks = []
    csv_path = SQND_DIR / "dear_abby_data" / "raw_da_qs.csv"

    if not csv_path.exists():
        log.warning(f"Dear Abby CSV not found: {csv_path}")
        return []

    log.info(f"Loading Dear Abby from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            text = row.get("question_only", "").strip()
            if not text or len(text) < 100:
                continue

            year = None
            try:
                year = int(float(row.get("year", "0")))
            except (ValueError, TypeError):
                pass

            century = (year // 100 + 1) if year and year > 0 else 20
            period = "MODERN"
            if year:
                if year < 1970:
                    period = "POSTWAR"
                elif year < 1990:
                    period = "LATE_20C"
                else:
                    period = "MODERN"

            title = row.get("title", "")
            ref = f"Dear Abby #{row.get('letterId', count)} ({year}): {title[:60]}"

            for chunk in chunk_text(text):
                chunks.append(EthicsChunk(
                    corpus="dear_abby",
                    tradition="american_advice",
                    language="english",
                    period=period,
                    century=century,
                    source_ref=ref,
                    content=chunk,
                ))
            count += 1
            if count >= 50000:
                break

    log.info(f"Dear Abby: {len(chunks)} chunks from {count} letters")
    return chunks


# ============================================================================
# Corpus 5: Perseus Digital Library (Greek/Latin)
# ============================================================================

def fetch_perseus() -> list[EthicsChunk]:
    """Clone Perseus canonical-greekLit and parse TEI XML."""
    corpus_dir = BASE_DIR / "perseus"
    greek_dir = corpus_dir / "canonical-greekLit"
    latin_dir = corpus_dir / "canonical-latinLit"
    chunks = []

    # Clone Greek texts
    if not greek_dir.exists():
        log.info("Cloning Perseus canonical-greekLit (--depth 1)...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/PerseusDL/canonical-greekLit.git",
             str(greek_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error(f"Perseus Greek clone failed: {result.stderr}")
        else:
            log.info("Perseus Greek clone complete.")

    # Clone Latin texts
    if not latin_dir.exists():
        log.info("Cloning Perseus canonical-latinLit (--depth 1)...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/PerseusDL/canonical-latinLit.git",
             str(latin_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error(f"Perseus Latin clone failed: {result.stderr}")
        else:
            log.info("Perseus Latin clone complete.")

    # Parse TEI XML files
    from lxml import etree

    # Greek authors with ethics content
    GREEK_ETHICS_AUTHORS = {
        "tlg0059": ("Plato", "ancient_greek", "CLASSICAL", -4),
        "tlg0086": ("Aristotle", "ancient_greek", "CLASSICAL", -4),
        "tlg0627": ("Hippocrates", "ancient_greek", "CLASSICAL", -5),
        "tlg0007": ("Plutarch", "ancient_greek", "IMPERIAL", 1),
        "tlg0032": ("Xenophon", "ancient_greek", "CLASSICAL", -4),
        "tlg4015": ("Epictetus", "ancient_greek", "IMPERIAL", 1),
        "tlg0006": ("Euripides", "ancient_greek", "CLASSICAL", -5),
        "tlg0011": ("Sophocles", "ancient_greek", "CLASSICAL", -5),
        "tlg0012": ("Homer", "ancient_greek", "ARCHAIC", -8),
        "tlg0085": ("Aeschylus", "ancient_greek", "CLASSICAL", -5),
        "tlg0062": ("Lucian", "ancient_greek", "IMPERIAL", 2),
        "tlg0557": ("Epictetus_Discourses", "ancient_greek", "IMPERIAL", 1),
        "tlg0613": ("Marcus_Aurelius", "ancient_greek", "IMPERIAL", 2),
    }

    LATIN_ETHICS_AUTHORS = {
        "phi0474": ("Cicero", "latin", "LATE_REPUBLIC", -1),
        "phi1351": ("Seneca", "latin", "IMPERIAL", 1),
        "phi0690": ("Lucretius", "latin", "LATE_REPUBLIC", -1),
        "phi0975": ("Virgil", "latin", "AUGUSTAN", -1),
        "phi0631": ("Horace", "latin", "AUGUSTAN", -1),
        "phi1345": ("Petronius", "latin", "IMPERIAL", 1),
        "phi0588": ("Sallust", "latin", "LATE_REPUBLIC", -1),
        "phi0978": ("Tacitus", "latin", "IMPERIAL", 2),
        "phi0448": ("Caesar", "latin", "LATE_REPUBLIC", -1),
    }

    for repo_dir, authors, tradition in [
        (greek_dir, GREEK_ETHICS_AUTHORS, "greco_roman"),
        (latin_dir, LATIN_ETHICS_AUTHORS, "greco_roman"),
    ]:
        if not repo_dir.exists():
            continue
        data_dir = repo_dir / "data"
        if not data_dir.exists():
            log.warning(f"No data dir in {repo_dir}")
            continue

        file_count = 0
        for author_id, (author_name, language, period, century) in authors.items():
            author_dir = data_dir / author_id
            if not author_dir.exists():
                # Try searching
                for d in data_dir.iterdir():
                    if d.is_dir() and author_id in d.name:
                        author_dir = d
                        break
                else:
                    continue

            xml_files = list(author_dir.rglob("*.xml"))
            for xml_file in xml_files[:50]:  # Limit per author
                try:
                    parsed_chunks = _parse_tei_xml(
                        xml_file, author_name, language, period, century, tradition
                    )
                    chunks.extend(parsed_chunks)
                    file_count += 1
                except Exception as e:
                    log.debug(f"Failed to parse {xml_file}: {e}")

        log.info(f"  Perseus {tradition}: {file_count} files, {len(chunks)} chunks so far")

    log.info(f"Perseus total: {len(chunks)} chunks")
    return chunks


def _parse_tei_xml(
    xml_file: Path, author: str, language: str, period: str, century: int, tradition: str
) -> list[EthicsChunk]:
    """Parse a TEI XML file and extract text chunks."""
    from lxml import etree

    chunks = []
    try:
        tree = etree.parse(str(xml_file))
    except etree.XMLSyntaxError:
        return []

    root = tree.getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # Try to get the title
    title_elem = root.find(".//tei:titleStmt/tei:title", ns)
    if title_elem is None:
        title_elem = root.find(".//{http://www.tei-c.org/ns/1.0}title")
    title = title_elem.text if title_elem is not None and title_elem.text else xml_file.stem

    # Extract text from body
    body = root.find(".//tei:body", ns)
    if body is None:
        body = root.find(".//{http://www.tei-c.org/ns/1.0}body")
    if body is None:
        return []

    # Get all text content from divs/paragraphs
    text_parts = []
    for elem in body.iter():
        if elem.text:
            text_parts.append(elem.text.strip())
        if elem.tail:
            text_parts.append(elem.tail.strip())

    full_text = " ".join(t for t in text_parts if t)
    if not full_text or len(full_text) < 100:
        return []

    for chunk_text_str in chunk_text(full_text):
        chunks.append(EthicsChunk(
            corpus="perseus",
            tradition=tradition,
            language=language,
            period=period,
            century=century,
            source_ref=f"{author}: {title[:80]}",
            content=chunk_text_str,
        ))

    return chunks


# ============================================================================
# Corpus 6: Pali Canon (SuttaCentral API)
# ============================================================================

SUTTA_COLLECTIONS = [
    # (collection, name, subcollections)
    ("dn", "Digha Nikaya", list(range(1, 35))),     # 34 suttas
    ("mn", "Majjhima Nikaya", list(range(1, 153))),  # 152 suttas
    ("sn", "Samyutta Nikaya", [f"{i}.{j}" for i in range(1, 57) for j in range(1, 12)]),
    ("an", "Anguttara Nikaya", [f"{i}.{j}" for i in range(1, 12) for j in range(1, 20)]),
    ("dhp", "Dhammapada", list(range(1, 27))),       # 26 chapters
]


def fetch_pali_canon() -> list[EthicsChunk]:
    """Fetch Pali Canon texts from SuttaCentral API."""
    corpus_dir = BASE_DIR / "pali_canon"
    cache_file = corpus_dir / "pali_texts.json"
    chunks = []

    if cache_file.exists():
        log.info(f"Using cached Pali data: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            all_texts = json.load(f)
    else:
        log.info("Fetching Pali Canon from SuttaCentral API...")
        all_texts = []
        fetched = 0
        max_fetch = 2000  # Limit to be respectful

        for collection, coll_name, ids in SUTTA_COLLECTIONS:
            log.info(f"  Fetching {coll_name}...")
            for sid in ids:
                if fetched >= max_fetch:
                    break
                sutta_id = f"{collection}{sid}"
                url = f"https://suttacentral.net/api/bilarasuttas/{sutta_id}/pli"
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "EthicsCorpora/1.0",
                        "Accept": "application/json",
                    })
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    # Extract Pali text from bilara format
                    root_text = data.get("root_text", {})
                    if root_text:
                        text_parts = []
                        for key in sorted(root_text.keys()):
                            text_parts.append(root_text[key])
                        full_text = " ".join(text_parts)
                        if full_text and len(full_text) >= 50:
                            all_texts.append({
                                "text": full_text,
                                "ref": f"{coll_name} {sid}",
                                "sutta_id": sutta_id,
                            })
                            fetched += 1

                    time.sleep(1.0)  # Respectful rate limit
                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    # Many IDs won't exist (sparse numbering in SN/AN)
                    pass
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    log.debug(f"  Sutta {sutta_id}: {e}")

            if fetched >= max_fetch:
                break
            log.info(f"  {coll_name}: {fetched} suttas fetched so far")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
        log.info(f"  Cached {len(all_texts)} Pali texts")

    for entry in all_texts:
        text = entry.get("text", "")
        for chunk in chunk_text(text):
            chunks.append(EthicsChunk(
                corpus="pali_canon",
                tradition="buddhist",
                language="pali",
                period="BUDDHIST_EARLY",
                century=-3,
                source_ref=entry.get("ref", ""),
                content=chunk,
            ))

    log.info(f"Pali Canon: {len(chunks)} chunks from {len(all_texts)} suttas")
    return chunks


# ============================================================================
# Corpus 7: UN Human Rights Declarations
# ============================================================================

UN_DOCS = [
    {
        "name": "Universal Declaration of Human Rights",
        "url": "https://www.un.org/en/about-us/universal-declaration-of-human-rights",
        "lang": "english",
        "year": 1948,
    },
]

# UDHR text is well-known; include it directly to avoid scraping issues
UDHR_TEXT = """
Article 1. All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.
Article 2. Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind, such as race, colour, sex, language, religion, political or other opinion, national or social origin, property, birth or other status. Furthermore, no distinction shall be made on the basis of the political, jurisdictional or international status of the country or territory to which a person belongs, whether it be independent, trust, non-self-governing or under any other limitation of sovereignty.
Article 3. Everyone has the right to life, liberty and security of person.
Article 4. No one shall be held in slavery or servitude; slavery and the slave trade shall be prohibited in all their forms.
Article 5. No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.
Article 6. Everyone has the right to recognition everywhere as a person before the law.
Article 7. All are equal before the law and are entitled without any discrimination to equal protection of the law. All are entitled to equal protection against any discrimination in violation of this Declaration and against any incitement to such discrimination.
Article 8. Everyone has the right to an effective remedy by the competent national tribunals for acts violating the fundamental rights granted him by the constitution or by law.
Article 9. No one shall be subjected to arbitrary arrest, detention or exile.
Article 10. Everyone is entitled in full equality to a fair and public hearing by an independent and impartial tribunal, in the determination of his rights and obligations and of any criminal charge against him.
Article 11. (1) Everyone charged with a penal offence has the right to be presumed innocent until proved guilty according to law in a public trial at which he has had all the guarantees necessary for his defence. (2) No one shall be held guilty of any penal offence on account of any act or omission which did not constitute a penal offence, under national or international law, at the time when it was committed. Nor shall a heavier penalty be imposed than the one that was applicable at the time the penal offence was committed.
Article 12. No one shall be subjected to arbitrary interference with his privacy, family, home or correspondence, nor to attacks upon his honour and reputation. Everyone has the right to the protection of the law against such interference or attacks.
Article 13. (1) Everyone has the right to freedom of movement and residence within the borders of each state. (2) Everyone has the right to leave any country, including his own, and to return to his country.
Article 14. (1) Everyone has the right to seek and to enjoy in other countries asylum from persecution. (2) This right may not be invoked in the case of prosecutions genuinely arising from non-political crimes or from acts contrary to the purposes and principles of the United Nations.
Article 15. (1) Everyone has the right to a nationality. (2) No one shall be arbitrarily deprived of his nationality nor denied the right to change his nationality.
Article 16. (1) Men and women of full age, without any limitation due to race, nationality or religion, have the right to marry and to found a family. They are entitled to equal rights as to marriage, during marriage and at its dissolution. (2) Marriage shall be entered into only with the free and full consent of the intending spouses. (3) The family is the natural and fundamental group unit of society and is entitled to protection by society and the State.
Article 17. (1) Everyone has the right to own property alone as well as in association with others. (2) No one shall be arbitrarily deprived of his property.
Article 18. Everyone has the right to freedom of thought, conscience and religion; this right includes freedom to change his religion or belief, and freedom, either alone or in community with others and in public or private, to manifest his religion or belief in teaching, practice, worship and observance.
Article 19. Everyone has the right to freedom of opinion and expression; this right includes freedom to hold opinions without interference and to seek, receive and impart information and ideas through any media and regardless of frontiers.
Article 20. (1) Everyone has the right to freedom of peaceful assembly and association. (2) No one may be compelled to belong to an association.
Article 21. (1) Everyone has the right to take part in the government of his country, directly or through freely chosen representatives. (2) Everyone has the right of equal access to public service in his country. (3) The will of the people shall be the basis of the authority of government; this will shall be expressed in periodic and genuine elections which shall be by universal and equal suffrage and shall be held by secret vote or by equivalent free voting procedures.
Article 22. Everyone, as a member of society, has the right to social security and is entitled to realization, through national effort and international co-operation and in accordance with the organization and resources of each State, of the economic, social and cultural rights indispensable for his dignity and the free development of his personality.
Article 23. (1) Everyone has the right to work, to free choice of employment, to just and favourable conditions of work and to protection against unemployment. (2) Everyone, without any discrimination, has the right to equal pay for equal work. (3) Everyone who works has the right to just and favourable remuneration ensuring for himself and his family an existence worthy of human dignity, and supplemented, if necessary, by other means of social protection. (4) Everyone has the right to form and to join trade unions for the protection of his interests.
Article 24. Everyone has the right to rest and leisure, including reasonable limitation of working hours and periodic holidays with pay.
Article 25. (1) Everyone has the right to a standard of living adequate for the health and well-being of himself and of his family, including food, clothing, housing and medical care and necessary social services, and the right to security in the event of unemployment, sickness, disability, widowhood, old age or other lack of livelihood in circumstances beyond his control. (2) Motherhood and childhood are entitled to special care and assistance. All children, whether born in or out of wedlock, shall enjoy the same social protection.
Article 26. (1) Everyone has the right to education. Education shall be free, at least in the elementary and fundamental stages. Elementary education shall be compulsory. Technical and professional education shall be made generally available and higher education shall be equally accessible to all on the basis of merit. (2) Education shall be directed to the full development of the human personality and to the strengthening of respect for human rights and fundamental freedoms. It shall promote understanding, tolerance and friendship among all nations, racial or religious groups, and shall further the activities of the United Nations for the maintenance of peace. (3) Parents have a prior right to choose the kind of education that shall be given to their children.
Article 27. (1) Everyone has the right freely to participate in the cultural life of the community, to enjoy the arts and to share in scientific advancement and its benefits. (2) Everyone has the right to the protection of the moral and material interests resulting from any scientific, literary or artistic production of which he is the author.
Article 28. Everyone is entitled to a social and international order in which the rights and freedoms set forth in this Declaration can be fully realized.
Article 29. (1) Everyone has duties to the community in which alone the free and full development of his personality is possible. (2) In the exercise of his rights and freedoms, everyone shall be subject only to such limitations as are determined by law solely for the purpose of securing due recognition and respect for the rights and freedoms of others and of meeting the just requirements of morality, public order and the general welfare in a democratic society. (3) These rights and freedoms may in no case be exercised contrary to the purposes and principles of the United Nations.
Article 30. Nothing in this Declaration may be interpreted as implying for any State, group or person any right to engage in any activity or to perform any act aimed at the destruction of any of the rights and freedoms set forth herein.
""".strip()


def fetch_un_declarations() -> list[EthicsChunk]:
    """Create chunks from UN Human Rights declarations."""
    chunks = []
    corpus_dir = BASE_DIR / "un_declarations"

    # UDHR English
    articles = re.split(r'(?=Article \d+\.)', UDHR_TEXT)
    for article in articles:
        article = article.strip()
        if not article or len(article) < 50:
            continue
        # Extract article number
        m = re.match(r'Article (\d+)\.', article)
        ref = f"UDHR Art. {m.group(1)}" if m else "UDHR"

        chunks.append(EthicsChunk(
            corpus="un_declarations",
            tradition="international_human_rights",
            language="english",
            period="MODERN",
            century=20,
            source_ref=ref,
            content=article,
        ))

    # Try to fetch UDHR in other languages from unicode.org UDHR collection
    udhr_langs = {
        "fra": ("french", "Declaration universelle des droits de l'homme"),
        "spa": ("spanish", "Declaracion Universal de Derechos Humanos"),
        "ara": ("arabic", "Universal Declaration - Arabic"),
        "cmn_Hans": ("chinese_simplified", "Universal Declaration - Chinese"),
        "rus": ("russian", "Universal Declaration - Russian"),
        "hin": ("hindi", "Universal Declaration - Hindi"),
    }

    for lang_code, (lang_name, desc) in udhr_langs.items():
        cache_file = corpus_dir / f"udhr_{lang_code}.txt"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            url = f"https://www.unicode.org/udhr/d/udhr_{lang_code}.txt"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "EthicsCorpora/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
                log.info(f"  Fetched UDHR in {lang_name}")
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  Failed to fetch UDHR {lang_code}: {e}")
                continue

        # Parse into articles
        # Unicode UDHR format has articles separated by blank lines
        paragraphs = re.split(r'\n\s*\n', text)
        art_num = 0
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 30:
                continue
            # Skip header lines
            if para.startswith("---") or para.startswith("Universal") or para.startswith("Note"):
                continue
            art_num += 1
            chunks.append(EthicsChunk(
                corpus="un_declarations",
                tradition="international_human_rights",
                language=lang_name,
                period="MODERN",
                century=20,
                source_ref=f"UDHR ({lang_name}) para {art_num}",
                content=para,
            ))

    log.info(f"UN Declarations: {len(chunks)} chunks")
    return chunks


# ============================================================================
# Embedding generation with BGE-M3
# ============================================================================

def generate_embeddings():
    """Generate BGE-M3 embeddings for all chunks missing embeddings."""
    log.info("Loading BGE-M3 model for embedding generation...")
    log.info("  Using CPU (CUDA_VISIBLE_DEVICES unset for thermal safety)")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    log.info("  BGE-M3 model loaded.")

    conn = get_db_conn()
    cur = conn.cursor()

    # Count total chunks needing embeddings
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NULL")
    total = cur.fetchone()[0]
    log.info(f"  {total} chunks need embeddings")

    if total == 0:
        cur.close()
        conn.close()
        return

    # Process in batches
    processed = 0
    batch_size = EMBEDDING_BATCH_SIZE

    while processed < total:
        # Fetch a batch of chunks without embeddings
        cur.execute(
            "SELECT id, content FROM ethics_chunks WHERE embedding IS NULL ORDER BY id LIMIT %s",
            (batch_size,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # Generate embeddings
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

        # Update database
        update_cur = conn.cursor()
        for chunk_id, emb in zip(ids, embeddings):
            emb_list = emb.tolist()
            update_cur.execute(
                "UPDATE ethics_chunks SET embedding = %s WHERE id = %s",
                (str(emb_list), chunk_id),
            )
        conn.commit()
        update_cur.close()

        processed += len(rows)
        elapsed_pct = (processed / total) * 100
        log.info(f"  Embedded {processed}/{total} ({elapsed_pct:.1f}%)")

    cur.close()
    conn.close()
    log.info(f"Embedding generation complete: {processed} chunks embedded")


# ============================================================================
# Create embedding index
# ============================================================================

def create_embedding_index():
    """Create IVFFlat index on the embedding column."""
    log.info("Creating IVFFlat index on ethics_chunks.embedding...")
    conn = get_db_conn()
    cur = conn.cursor()

    # Count total rows for optimal list count
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE embedding IS NOT NULL")
    total = cur.fetchone()[0]

    if total == 0:
        log.info("  No embedded chunks yet, skipping index creation.")
        cur.close()
        conn.close()
        return

    # IVFFlat: lists ~ sqrt(n)
    import math
    n_lists = max(10, min(1000, int(math.sqrt(total))))

    cur.execute("DROP INDEX IF EXISTS idx_ethics_embedding;")
    cur.execute(f"""
        CREATE INDEX idx_ethics_embedding ON ethics_chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = {n_lists});
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"  IVFFlat index created with {n_lists} lists for {total} rows")


# ============================================================================
# Statistics
# ============================================================================

def print_stats():
    """Print summary statistics."""
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM ethics_chunks")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT corpus, tradition, language, count(*),
               count(embedding), min(century), max(century)
        FROM ethics_chunks
        GROUP BY corpus, tradition, language
        ORDER BY corpus
    """)
    rows = cur.fetchall()

    log.info("=" * 80)
    log.info(f"ETHICS CORPORA SUMMARY: {total} total chunks")
    log.info("=" * 80)
    log.info(f"{'Corpus':<20} {'Tradition':<25} {'Language':<20} {'Chunks':>8} {'Embedded':>8} {'Period':>12}")
    log.info("-" * 95)
    for corpus, tradition, language, count, embedded, min_c, max_c in rows:
        period = f"{min_c or '?'}c-{max_c or '?'}c"
        log.info(f"{corpus:<20} {tradition:<25} {language:<20} {count:>8} {embedded:>8} {period:>12}")
    log.info("=" * 80)

    cur.close()
    conn.close()


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    log.info("=" * 80)
    log.info("ETHICS CORPORA PIPELINE - Starting")
    log.info("=" * 80)

    # Ensure base directory
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Create table
    create_table()

    # Phase 1: Fetch and index corpora (text only)
    corpora_fetchers = [
        ("dear_abby", fetch_dear_abby),           # Quick - local CSV
        ("un_declarations", fetch_un_declarations), # Quick - embedded text + small downloads
        ("sefaria", fetch_sefaria),                # Large git clone ~2GB
        ("chinese_classics", fetch_chinese_classics), # API fetch (slow, rate-limited)
        ("islamic", fetch_islamic),                # API fetch
        ("perseus", fetch_perseus),                # Git clone ~1GB
        ("pali_canon", fetch_pali_canon),          # API fetch (slow, rate-limited)
    ]

    for corpus_name, fetcher in corpora_fetchers:
        existing = count_corpus_rows(corpus_name)
        if existing > 0:
            log.info(f"SKIP {corpus_name}: already has {existing} chunks in database")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"FETCHING: {corpus_name}")
        log.info(f"{'='*60}")

        try:
            chunks = fetcher()
            if chunks:
                inserted = insert_chunks(chunks)
                log.info(f"DONE {corpus_name}: inserted {inserted} chunks")
            else:
                log.warning(f"EMPTY {corpus_name}: no chunks produced")
        except Exception as e:
            log.error(f"FAILED {corpus_name}: {e}", exc_info=True)

    # Print stats before embedding
    print_stats()

    # Phase 2: Generate embeddings
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: Generating BGE-M3 embeddings")
    log.info("=" * 60)

    try:
        generate_embeddings()
    except Exception as e:
        log.error(f"Embedding generation failed: {e}", exc_info=True)

    # Phase 3: Create index
    try:
        create_embedding_index()
    except Exception as e:
        log.error(f"Index creation failed: {e}", exc_info=True)

    # Final stats
    print_stats()

    log.info("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
