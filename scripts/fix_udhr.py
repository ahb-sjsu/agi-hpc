#!/usr/bin/env python3
"""Fix UDHR multilingual - parse NLTK UDHR2 corpus and insert into ethics_chunks."""

import zipfile
import re

import psycopg2
import psycopg2.extras

LANG_MAP = {
    'arb': 'arabic', 'cmn': 'chinese_mandarin', 'fra': 'french',
    'deu': 'german', 'hin': 'hindi', 'jpn': 'japanese',
    'kor': 'korean', 'rus': 'russian', 'spa': 'spanish',
    'swa': 'swahili', 'tur': 'turkish', 'heb': 'hebrew',
    'por': 'portuguese', 'ita': 'italian', 'ell': 'greek',
    'tha': 'thai', 'vie': 'vietnamese', 'ind': 'indonesian',
    'msa': 'malay', 'urd': 'urdu', 'fas': 'persian',
    'ben': 'bengali', 'tam': 'tamil', 'tel': 'telugu',
    'pol': 'polish', 'ukr': 'ukrainian', 'ron': 'romanian',
    'hun': 'hungarian', 'ces': 'czech', 'nld': 'dutch',
    'swe': 'swedish', 'dan': 'danish',
    'fin': 'finnish', 'kat': 'georgian', 'hye': 'armenian',
    'amh': 'amharic', 'zul': 'zulu', 'yor': 'yoruba',
    'hau': 'hausa', 'ibo': 'igbo',
}

zf = zipfile.ZipFile('/archive/ethics-corpora/un_declarations/udhr2.zip')
chunks = []
DOUBLE_NEWLINE = "\n\n"

for name in sorted(zf.namelist()):
    if not name.endswith('.txt'):
        continue
    base = name.split('/')[-1].replace('.txt', '')
    lang = LANG_MAP.get(base)
    if not lang:
        continue

    try:
        text = zf.read(name).decode('utf-8', errors='replace')
    except Exception:
        continue

    paragraphs = text.split(DOUBLE_NEWLINE)
    art_num = 0
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 30:
            continue
        art_num += 1
        chunks.append((
            'un_declarations',
            'international_human_rights',
            lang,
            'MODERN',
            20,
            f'UDHR ({lang}) para {art_num}',
            para[:1000],
        ))

zf.close()

print(f'UDHR chunks to insert: {len(chunks)}')
langs = sorted(set(c[2] for c in chunks))
print(f'Languages ({len(langs)}): {langs}')

if chunks:
    conn = psycopg2.connect(dbname='atlas', user='claude')
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO ethics_chunks
           (corpus, tradition, language, period, century, source_ref, content)
           VALUES %s""",
        chunks,
        template='(%s, %s, %s, %s, %s, %s, %s)',
    )
    conn.commit()
    print(f'Inserted {len(chunks)} UDHR multilingual chunks')
    cur.close()
    conn.close()
