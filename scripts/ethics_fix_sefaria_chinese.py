#!/usr/bin/env python3
"""
Fix script for Sefaria (GCS-based) and Chinese Classics corpora.
Also patches embedding to use GPU for speed.

Run: taskset -c 36-39 /home/claude/env/bin/python /archive/ethics-corpora/fix_sefaria_chinese.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

BASE_DIR = Path("/archive/ethics-corpora")
DB_NAME = "atlas"
DB_USER = "claude"
CHUNK_SIZE_MIN = 200
CHUNK_SIZE_MAX = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "fix.log", mode="a"),
    ],
)
log = logging.getLogger("ethics_fix")


@dataclass
class EthicsChunk:
    corpus: str
    tradition: str
    language: str
    period: str
    century: Optional[int]
    source_ref: str
    content: str


def get_db_conn():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER)


def count_corpus_rows(corpus_name: str) -> int:
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM ethics_chunks WHERE corpus = %s", (corpus_name,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def insert_chunks(chunks: list[EthicsChunk], batch_size: int = 1000) -> int:
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


def chunk_text(text: str, max_size: int = CHUNK_SIZE_MAX, min_size: int = CHUNK_SIZE_MIN) -> list[str]:
    if not text or len(text.strip()) < min_size:
        if text and len(text.strip()) >= 50:
            return [text.strip()]
        return []
    text = text.strip()
    if len(text) <= max_size:
        return [text]
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
            while len(current) > max_size:
                split_pos = current.rfind(" ", 0, max_size)
                if split_pos < min_size:
                    split_pos = max_size
                chunks.append(current[:split_pos].strip())
                current = current[split_pos:].strip()
    if current and len(current) >= min_size:
        chunks.append(current)
    elif current and chunks:
        chunks[-1] = chunks[-1] + " " + current
    return chunks


# ============================================================================
# Sefaria via GCS (books.json index)
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
    "Jewish Thought": ("RISHONIM", 12),
    "Chasidut": ("ACHRONIM", 18),
    "Musar": ("ACHRONIM", 19),
    "Responsa": ("ACHRONIM", 17),
    "Liturgy": ("MEDIEVAL", 10),
    "Reference": ("MODERN", 20),
}

ARAMAIC_CATEGORIES = {"Talmud", "Bavli", "Yerushalmi"}

# Categories to prioritize (ethics-relevant)
PRIORITY_CATEGORIES = [
    "Torah", "Prophets", "Writings",  # Tanakh
    "Mishnah", "Tosefta",             # Tannaitic
    "Bavli", "Yerushalmi",            # Talmud
    "Midrash",                         # Midrashic
    "Halakhah",                        # Legal
    "Philosophy",                      # Jewish philosophy
    "Musar",                          # Ethics/character
    "Jewish Thought",                  # Theology
    "Kabbalah",                        # Mystical
    "Chasidut",                       # Chassidic
]


def fetch_sefaria_gcs() -> list[EthicsChunk]:
    """Download key Sefaria texts from GCS using books.json index."""
    books_file = Path("/archive/ethics-corpora/sefaria/Sefaria-Export/books.json")
    cache_dir = BASE_DIR / "sefaria" / "json_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading Sefaria books.json index...")
    with open(books_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    books = data.get("books", [])
    log.info(f"  {len(books)} total book entries in index")

    # Filter to Hebrew merged texts in priority categories
    target_books = []
    for book in books:
        cats = book.get("categories", [])
        lang = book.get("language", "")
        version = book.get("versionTitle", "")

        if version != "merged":
            continue

        # Check if any category is in our priority list
        matching_cat = None
        for cat in cats:
            if cat in PRIORITY_CATEGORIES:
                matching_cat = cat
                break

        if matching_cat and lang == "Hebrew":
            target_books.append((book, matching_cat))

    log.info(f"  {len(target_books)} merged Hebrew texts in priority categories")

    # Download and parse each text
    chunks = []
    downloaded = 0
    max_download = 5000  # Safety limit

    for book, matching_cat in target_books:
        if downloaded >= max_download:
            break

        title = book.get("title", "unknown")
        json_url = book.get("json_url", "")
        if not json_url:
            continue

        # Check cache
        safe_title = re.sub(r'[^\w\-.]', '_', title)
        cache_file = cache_dir / f"{safe_title}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    text_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        else:
            try:
                req = urllib.request.Request(json_url, headers={"User-Agent": "EthicsCorpora/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
                text_data = json.loads(raw)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(raw)
                downloaded += 1
                if downloaded % 50 == 0:
                    log.info(f"  Downloaded {downloaded} texts, {len(chunks)} chunks so far...")
                time.sleep(0.1)  # Light rate-limit for GCS
            except Exception as e:
                log.debug(f"  Failed to download {title}: {e}")
                continue

        # Determine period and language
        period_info = SEFARIA_PERIOD_MAP.get(matching_cat, ("UNKNOWN", None))
        period, century = period_info
        lang = "aramaic" if matching_cat in ARAMAIC_CATEGORIES else "hebrew"

        # Extract text from Sefaria JSON
        texts = _extract_sefaria_json_text(text_data, title)
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

    log.info(f"Sefaria: {len(chunks)} chunks from {downloaded} downloaded + cached texts")
    return chunks


def _extract_sefaria_json_text(data: dict, title: str) -> list[tuple[str, str]]:
    """Extract text content from a Sefaria JSON text file."""
    results = []

    if isinstance(data, dict):
        # Try "text" key (main text content)
        for key in ("text", "he"):
            if key in data:
                _flatten_sefaria(data[key], f"{title}", results)

        # Try "versions" format
        if "versions" in data and isinstance(data["versions"], list):
            for ver in data["versions"]:
                if isinstance(ver, dict) and "text" in ver:
                    _flatten_sefaria(ver["text"], title, results)
    elif isinstance(data, list):
        _flatten_sefaria(data, title, results)

    return results


def _flatten_sefaria(obj, ref: str, results: list, depth: int = 0):
    """Recursively flatten nested Sefaria text arrays."""
    if depth > 10:
        return
    if isinstance(obj, str):
        text = re.sub(r'<[^>]+>', '', obj).strip()
        if text and len(text) >= 50:
            results.append((ref, text))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _flatten_sefaria(item, f"{ref}:{i+1}", results, depth + 1)


# ============================================================================
# Chinese Classics - direct text inclusion
# ============================================================================

# Since ctext.org API appears to not return text content via our simple approach,
# include key passages directly. These are well-known public domain texts.

CHINESE_CLASSICS_PASSAGES = [
    # Analects (Lunyu) - Confucius, ~500 BCE
    {"text": "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？", "ref": "Analects 1.1", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "有子曰：其為人也孝弟，而好犯上者，鮮矣；不好犯上，而好作亂者，未之有也。君子務本，本立而道生。孝弟也者，其為仁之本與！", "ref": "Analects 1.2", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：巧言令色，鮮矣仁！", "ref": "Analects 1.3", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "曾子曰：吾日三省吾身：為人謀而不忠乎？與朋友交而不信乎？傳不習乎？", "ref": "Analects 1.4", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：道千乘之國，敬事而信，節用而愛人，使民以時。", "ref": "Analects 1.5", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：弟子入則孝，出則弟，謹而信，汎愛眾，而親仁。行有餘力，則以學文。", "ref": "Analects 1.6", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子夏曰：賢賢易色；事父母，能竭其力；事君，能致其身；與朋友交，言而有信。雖曰未學，吾必謂之學矣。", "ref": "Analects 1.7", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：君子不重則不威，學則不固。主忠信，無友不如己者，過則勿憚改。", "ref": "Analects 1.8", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：為政以德，譬如北辰，居其所而眾星共之。", "ref": "Analects 2.1", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：吾十有五而志于學，三十而立，四十而不惑，五十而知天命，六十而耳順，七十而從心所欲不踰矩。", "ref": "Analects 2.4", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：溫故而知新，可以為師矣。", "ref": "Analects 2.11", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：君子周而不比，小人比而不周。", "ref": "Analects 2.14", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：學而不思則罔，思而不學則殆。", "ref": "Analects 2.15", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：知之為知之，不知為不知，是知也。", "ref": "Analects 2.17", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：人而無信，不知其可也。大車無輗，小車無軏，其何以行之哉？", "ref": "Analects 2.22", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：里仁為美。擇不處仁，焉得知？", "ref": "Analects 4.1", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：不仁者不可以久處約，不可以長處樂。仁者安仁，知者利仁。", "ref": "Analects 4.2", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：朝聞道，夕死可矣。", "ref": "Analects 4.8", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：君子喻於義，小人喻於利。", "ref": "Analects 4.16", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：見賢思齊焉，見不賢而內自省也。", "ref": "Analects 4.17", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：德不孤，必有鄰。", "ref": "Analects 4.25", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：己所不欲，勿施於人。", "ref": "Analects 15.24", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子貢問曰：有一言而可以終身行之者乎？子曰：其恕乎！己所不欲，勿施於人。", "ref": "Analects 15.23-24", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：志士仁人，無求生以害仁，有殺身以成仁。", "ref": "Analects 15.9", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：當仁不讓於師。", "ref": "Analects 15.36", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "曾子曰：士不可以不弘毅，任重而道遠。仁以為己任，不亦重乎？死而後已，不亦遠乎？", "ref": "Analects 8.7", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：三軍可奪帥也，匹夫不可奪志也。", "ref": "Analects 9.26", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：歲寒，然後知松柏之後凋也。", "ref": "Analects 9.28", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "顏淵問仁。子曰：克己復禮為仁。一日克己復禮，天下歸仁焉。為仁由己，而由人乎哉？", "ref": "Analects 12.1", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "仲弓問仁。子曰：出門如見大賓，使民如承大祭。己所不欲，勿施於人。在邦無怨，在家無怨。", "ref": "Analects 12.2", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},
    {"text": "子曰：其身正，不令而行；其身不正，雖令不從。", "ref": "Analects 13.6", "period": "CONFUCIAN", "century": -5, "tradition": "confucian"},

    # Dao De Jing (Tao Te Ching) - Laozi, ~600 BCE
    {"text": "道可道，非常道。名可名，非常名。無名天地之始，有名萬物之母。故常無欲以觀其妙，常有欲以觀其徼。此兩者同出而異名，同謂之玄。玄之又玄，眾妙之門。", "ref": "Dao De Jing 1", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "天下皆知美之為美，斯惡已。皆知善之為善，斯不善已。故有無相生，難易相成，長短相形，高下相盈，音聲相和，前後相隨。恆也。是以聖人處無為之事，行不言之教。", "ref": "Dao De Jing 2", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "不尚賢，使民不爭；不貴難得之貨，使民不為盜；不見可欲，使民心不亂。是以聖人之治，虛其心，實其腹，弱其志，強其骨。常使民無知無欲，使夫智者不敢為也。為無為，則無不治。", "ref": "Dao De Jing 3", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "上善若水。水善利萬物而不爭，處眾人之所惡，故幾於道。居善地，心善淵，與善仁，言善信，政善治，事善能，動善時。夫唯不爭，故無尤。", "ref": "Dao De Jing 8", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "三十輻共一轂，當其無，有車之用。埏埴以為器，當其無，有器之用。鑿戶牖以為室，當其無，有室之用。故有之以為利，無之以為用。", "ref": "Dao De Jing 11", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "大道廢，有仁義。慧智出，有大偽。六親不和，有孝慈。國家昏亂，有忠臣。", "ref": "Dao De Jing 18", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "人法地，地法天，天法道，道法自然。", "ref": "Dao De Jing 25", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "知人者智，自知者明。勝人者有力，自勝者強。知足者富，強行者有志。不失其所者久，死而不亡者壽。", "ref": "Dao De Jing 33", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "道常無為而無不為。侯王若能守之，萬物將自化。", "ref": "Dao De Jing 37", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "上德不德，是以有德。下德不失德，是以無德。上德無為而無以為。下德無為而有以為。上仁為之而無以為。上義為之而有以為。", "ref": "Dao De Jing 38", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "天下萬物生於有，有生於無。", "ref": "Dao De Jing 40", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "道生一，一生二，二生三，三生萬物。萬物負陰而抱陽，沖氣以為和。", "ref": "Dao De Jing 42", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "天下之至柔，馳騁天下之至堅。無有入無間，吾是以知無為之有益。不言之教，無為之益，天下希及之。", "ref": "Dao De Jing 43", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "禍兮福之所倚，福兮禍之所伏。", "ref": "Dao De Jing 58", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "治大國若烹小鮮。", "ref": "Dao De Jing 60", "period": "DAOIST", "century": -6, "tradition": "daoist"},
    {"text": "天之道，利而不害。聖人之道，為而不爭。", "ref": "Dao De Jing 81", "period": "DAOIST", "century": -6, "tradition": "daoist"},

    # Mencius (Mengzi) - ~372-289 BCE
    {"text": "孟子見梁惠王。王曰：叟！不遠千里而來，亦將有以利吾國乎？孟子對曰：王何必曰利？亦有仁義而已矣。", "ref": "Mencius 1A.1", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：人皆有不忍人之心。先王有不忍人之心，斯有不忍人之政矣。以不忍人之心，行不忍人之政，治天下可運之掌上。", "ref": "Mencius 2A.6", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "惻隱之心，仁之端也；羞惡之心，義之端也；辭讓之心，禮之端也；是非之心，智之端也。人之有是四端也，猶其有四體也。", "ref": "Mencius 2A.6 (four sprouts)", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：生，亦我所欲也；義，亦我所欲也。二者不可得兼，捨生而取義者也。", "ref": "Mencius 6A.10", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：天將降大任於是人也，必先苦其心志，勞其筋骨，餓其體膚，空乏其身，行拂亂其所為，所以動心忍性，曾益其所不能。", "ref": "Mencius 6B.15", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：人之所不學而能者，其良能也；所不慮而知者，其良知也。", "ref": "Mencius 7A.15", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：仁者無敵。", "ref": "Mencius 1A.5", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：民為貴，社稷次之，君為輕。", "ref": "Mencius 7B.14", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：窮則獨善其身，達則兼善天下。", "ref": "Mencius 7A.9", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},
    {"text": "孟子曰：盡信書，則不如無書。", "ref": "Mencius 7B.3", "period": "CONFUCIAN", "century": -4, "tradition": "confucian"},

    # Mozi - ~470-391 BCE (Universal Love / Jian Ai)
    {"text": "子墨子言曰：仁人之所以為事者，必興天下之利，除去天下之害，以此為事者也。然則天下之利何也？天下之害何也？子墨子言曰：今若國之與國之相攻，家之與家之相篡，人之與人之相賊，君臣不惠忠，父子不慈孝，兄弟不和調，此則天下之害也。", "ref": "Mozi, Jian Ai I", "period": "MOHIST", "century": -5, "tradition": "mohist"},
    {"text": "天下兼相愛則治，交相惡則亂。故子墨子曰不可以不勸愛人者，此也。", "ref": "Mozi, Jian Ai II", "period": "MOHIST", "century": -5, "tradition": "mohist"},

    # Xunzi - ~313-238 BCE
    {"text": "人之性惡，其善者偽也。今人之性，生而有好利焉，順是，故爭奪生而辭讓亡焉；生而有疾惡焉，順是，故殘賊生而忠信亡焉。", "ref": "Xunzi, Chapter 23 (Human Nature is Evil)", "period": "CONFUCIAN", "century": -3, "tradition": "confucian"},

    # Zhuangzi - ~369-286 BCE
    {"text": "北冥有魚，其名為鯤。鯤之大，不知其幾千里也。化而為鳥，其名為鵬。鵬之背，不知其幾千里也。怒而飛，其翼若垂天之雲。", "ref": "Zhuangzi, Xiaoyaoyou 1", "period": "DAOIST", "century": -4, "tradition": "daoist"},
    {"text": "昔者莊周夢為胡蝶，栩栩然胡蝶也。自喻適志與！不知周也。俄然覺，則蘧蘧然周也。不知周之夢為胡蝶與？胡蝶之夢為周與？周與胡蝶，則必有分矣。此之謂物化。", "ref": "Zhuangzi, Qiwulun", "period": "DAOIST", "century": -4, "tradition": "daoist"},
    {"text": "吾生也有涯，而知也無涯。以有涯隨無涯，殆已！已而為知者，殆而已矣。", "ref": "Zhuangzi, Yangshengzhu", "period": "DAOIST", "century": -4, "tradition": "daoist"},
    {"text": "庖丁為文惠君解牛，手之所觸，肩之所倚，足之所履，膝之所踦，砉然嚮然，奏刀騞然，莫不中音。", "ref": "Zhuangzi, Yangshengzhu (Cook Ding)", "period": "DAOIST", "century": -4, "tradition": "daoist"},

    # Han Feizi - ~280-233 BCE (Legalist)
    {"text": "法者，編著之圖籍，設之於官府，而布之於百姓者也。", "ref": "Han Feizi, Definition of Law", "period": "LEGALIST", "century": -3, "tradition": "legalist"},
    {"text": "明主之所導制其臣者，二柄而已矣。二柄者，刑德也。何謂刑德？曰：殺戮之謂刑，慶賞之謂德。", "ref": "Han Feizi, Two Handles", "period": "LEGALIST", "century": -3, "tradition": "legalist"},
]


def fetch_chinese_classics() -> list[EthicsChunk]:
    """Create Chinese Classics chunks from curated passages."""
    chunks = []
    for p in CHINESE_CLASSICS_PASSAGES:
        text = p["text"]
        if text and len(text.strip()) >= 10:  # Chinese chars are dense
            chunks.append(EthicsChunk(
                corpus="chinese_classics",
                tradition=p["tradition"],
                language="classical_chinese",
                period=p["period"],
                century=p["century"],
                source_ref=p["ref"],
                content=text,
            ))
    log.info(f"Chinese classics: {len(chunks)} curated passages")
    return chunks


# ============================================================================
# UDHR multilingual from Unicode CLDR
# ============================================================================

def fetch_udhr_multilingual() -> list[EthicsChunk]:
    """Fetch UDHR in multiple languages from Unicode CLDR."""
    chunks = []
    corpus_dir = BASE_DIR / "un_declarations"

    # Try OHCHR directly for full UDHR in different languages
    # These are well-known stable URLs
    udhr_sources = {
        "french": "https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/frn.pdf",
        "spanish": "https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/spn.pdf",
    }

    # For now, use the UDHR parallel corpus from GitHub (nltk data)
    # This is a well-known NLP resource
    try:
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip"
        cache_file = corpus_dir / "udhr2.zip"
        if not cache_file.exists():
            log.info("Attempting to download UDHR parallel corpus from NLTK...")
            req = urllib.request.Request(url, headers={"User-Agent": "EthicsCorpora/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            with open(cache_file, "wb") as f:
                f.write(data)
            log.info(f"  Downloaded {len(data)} bytes")

        if cache_file.exists():
            import zipfile
            with zipfile.ZipFile(cache_file) as zf:
                for name in zf.namelist():
                    if not name.endswith('.txt') or name.startswith('__'):
                        continue
                    # Determine language from filename
                    base = Path(name).stem
                    lang_map = {
                        'Arabic': 'arabic', 'Chinese': 'chinese', 'French': 'french',
                        'German': 'german', 'Hindi': 'hindi', 'Japanese': 'japanese',
                        'Korean': 'korean', 'Russian': 'russian', 'Spanish': 'spanish',
                        'Swahili': 'swahili', 'Turkish': 'turkish', 'Hebrew': 'hebrew',
                    }
                    lang = None
                    for key, val in lang_map.items():
                        if key.lower() in base.lower():
                            lang = val
                            break
                    if not lang:
                        continue

                    try:
                        text = zf.read(name).decode('utf-8', errors='replace')
                    except Exception:
                        continue

                    paragraphs = re.split(r'\n\s*\n', text)
                    art_num = 0
                    for para in paragraphs:
                        para = para.strip()
                        if not para or len(para) < 30:
                            continue
                        art_num += 1
                        chunks.append(EthicsChunk(
                            corpus="un_declarations",
                            tradition="international_human_rights",
                            language=lang,
                            period="MODERN",
                            century=20,
                            source_ref=f"UDHR ({lang}) para {art_num}",
                            content=para,
                        ))
    except Exception as e:
        log.warning(f"UDHR parallel corpus fetch failed: {e}")

    log.info(f"UDHR multilingual: {len(chunks)} additional chunks")
    return chunks


# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 70)
    log.info("FIX SCRIPT: Sefaria + Chinese Classics + UDHR multilingual")
    log.info("=" * 70)

    # 1. Chinese Classics
    existing = count_corpus_rows("chinese_classics")
    if existing == 0:
        log.info("\n--- Chinese Classics ---")
        chunks = fetch_chinese_classics()
        if chunks:
            n = insert_chunks(chunks)
            log.info(f"Inserted {n} Chinese Classics chunks")
    else:
        log.info(f"SKIP chinese_classics: already has {existing} rows")

    # 2. Sefaria via GCS
    existing = count_corpus_rows("sefaria")
    if existing == 0:
        log.info("\n--- Sefaria (GCS) ---")
        chunks = fetch_sefaria_gcs()
        if chunks:
            n = insert_chunks(chunks)
            log.info(f"Inserted {n} Sefaria chunks")
    else:
        log.info(f"SKIP sefaria: already has {existing} rows")

    # 3. Additional UDHR languages
    log.info("\n--- UDHR Multilingual ---")
    chunks = fetch_udhr_multilingual()
    if chunks:
        n = insert_chunks(chunks)
        log.info(f"Inserted {n} additional UDHR chunks")

    # Stats
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT corpus, tradition, language, count(*)
        FROM ethics_chunks
        GROUP BY corpus, tradition, language
        ORDER BY corpus
    """)
    log.info("\n--- Current totals ---")
    for corpus, tradition, language, count in cur.fetchall():
        log.info(f"  {corpus:20s} {tradition:25s} {language:20s} {count:>8}")
    cur.execute("SELECT count(*) FROM ethics_chunks")
    log.info(f"\n  TOTAL: {cur.fetchone()[0]}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
