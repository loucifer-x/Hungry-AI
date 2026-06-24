"""
rag.py — PostgreSQL + pgvector RAG engine
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional, Tuple

import config
import psycopg2

logger = logging.getLogger(__name__)

_embedder: Optional["SentenceTransformer"] = None
_conn = None


# ─────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────

def get_chroma_collection():
    """Kept for backward compatibility — returns a PostgreSQL connection."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(
            dbname="ragdb",
            user="raguser",
            password="ragpass",
            host="localhost",
            port=5432,
        )
    return _conn


# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


# ─────────────────────────────────────────────
# CATEGORY SYSTEM
# ─────────────────────────────────────────────

RULES = [
    # ── Linux ─────────────────────────────────────────────────────────────────
    ("linux", "commands", 10, ["ls", "cd", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "cat", "echo"]),

    ("other", "other", 0, []),
]

# ─────────────────────────────────────────────
# Precomputed lookup: keyword → (cat, sub, base_score)
#
# Keywords are split into two buckets:
#   PHRASE keywords  — contain a space or are >5 chars (more specific, fewer
#                      false-positives).  Matched as substrings, worth 2×.
#   TOKEN keywords   — short, single words (≤5 chars).  Must match as whole
#                      words (word-boundary check) to avoid "sh" hitting
#                      "phishing", "cat" hitting "category", etc.  Worth 1×.
#
# Built once at import time.
# ─────────────────────────────────────────────

# Short tokens that are far too common in natural-language source names to be
# reliable category signals on their own (e.g. "sh" inside "phishing").
_AMBIGUOUS_TOKENS = {
    "sh", "cat", "rm", "mv", "cp", "ls", "cd", "ps", "ip",
    "ss", "ts", "go", ".go", "rs", "cs", "js",
}

_PHRASE_INDEX: dict[str, list[tuple[str, str, int]]] = defaultdict(list)  # multi-word / long keywords
_TOKEN_INDEX:  dict[str, list[tuple[str, str, int]]] = defaultdict(list)  # short single-word keywords

for _cat, _sub, _score, _kws in RULES:
    for _kw in _kws:
        if _kw in _AMBIGUOUS_TOKENS:
            continue  # drop known noise tokens entirely
        if " " in _kw or len(_kw) > 5:
            _PHRASE_INDEX[_kw].append((_cat, _sub, _score))
        else:
            _TOKEN_INDEX[_kw].append((_cat, _sub, _score))

# Combined index used by is_retrieval_query (doesn't need the split)
_KEYWORD_INDEX = {**_PHRASE_INDEX, **_TOKEN_INDEX}

# Pre-compile word-boundary patterns for short token matching.
# Filenames use underscores and hyphens as separators, so we treat them as
# non-alphanumeric boundaries alongside standard whitespace / punctuation.
_TOKEN_PATTERN_CACHE: dict[str, re.Pattern] = {}

def _token_re(token: str) -> re.Pattern:
    if token not in _TOKEN_PATTERN_CACHE:
        # A "word boundary" here means: not preceded/followed by a letter or digit.
        # This correctly handles:
        #   "nmap_tutorial"   → "nmap" matches (boundary = "_")
        #   "phishing"        → "sh"   does NOT match (preceded by "i")
        #   "aes_key"         → "aes"  matches (boundary = "_")
        _TOKEN_PATTERN_CACHE[token] = re.compile(
            r"(?<![a-zA-Z0-9])" + re.escape(token) + r"(?![a-zA-Z0-9])",
            re.IGNORECASE,
        )
    return _TOKEN_PATTERN_CACHE[token]


def infer_category(source: str) -> Tuple[str, str]:
    """
    Score every rule against *source* and return the (category, subcategory)
    with the highest cumulative score.

    Scoring:
      - Multi-word / long phrases match as substrings and score 2× base.
      - Short single-word tokens must match at a word boundary and score 1× base.
      - Ties broken by subcategory string length (more specific sub wins).
    """
    s = source.lower()
    tally: dict[tuple[str, str], float] = defaultdict(float)

    # Phrase matches (substring — already specific enough)
    for keyword, entries in _PHRASE_INDEX.items():
        if keyword in s:
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 2.0

    # Token matches (whole-word only)
    for keyword, entries in _TOKEN_INDEX.items():
        if _token_re(keyword).search(s):
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 1.0

    if not tally:
        return "other", "other"

    best_cat, best_sub = max(
        tally,
        key=lambda k: (tally[k], len(k[1])),
    )
    return best_cat, best_sub


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:] + "\n\n" + p
            else:
                chunks.append(p)
                current = p[-overlap:]
    if current:
        chunks.append(current)
    return chunks


# ─────────────────────────────────────────────
# UPSERT
# ─────────────────────────────────────────────

def upsert_chunks(chunks: list[str], source: str, collection=None) -> int:
    if not chunks:
        return 0

    conn = get_chroma_collection()
    cur = conn.cursor()
    embedder = get_embedder()
    category, subcategory = infer_category(source)
    embeddings = embedder.encode(chunks).tolist()

    try:
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO documents (source, chunk_index, text, category, subcategory, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (source, i, text, category, subcategory, vec),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return len(chunks)


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

RAG_MIN_LENGTH = 10

RAG_SKIP_PHRASES = {
    "hi", "hello", "hey", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "whats up",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "what is your name",
}


def is_retrieval_query(query: str) -> bool:
    """
    Return True when the query is substantive enough to benefit from RAG.

    Logic (in order):
      1. Skip trivially short queries.
      2. Skip known small-talk phrases.
      3. Accept any query that scores above zero against the keyword index —
         this replaces the old exact-word-match loop which missed most real
         queries (e.g. "how do I use nmap?" never matched because "how" isn't
         a keyword, so the loop exited before reaching "nmap").
    """
    q = query.strip().lower()

    if len(q) < RAG_MIN_LENGTH:
        return False

    if q in RAG_SKIP_PHRASES:
        return False

    # Accept if any keyword from any rule appears anywhere in the query.
    for keyword in _KEYWORD_INDEX:
        if keyword in q:
            return True

    # Fallback: treat longer free-form questions as retrieval candidates even
    # when no explicit keyword matches — they may still hit relevant chunks via
    # semantic similarity.
    return len(q.split()) >= 5


def retrieve_context(
    query: str,
    top_k: int = config.TOP_K_RESULTS,
    min_score: float = config.MIN_RELEVANCE_SCORE,
):
    if not is_retrieval_query(query):
        return []

    embedder = get_embedder()
    qvec = embedder.encode(query).tolist()

    conn = get_chroma_collection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index,
                   embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, top_k),
        )
        rows = cur.fetchall()
    except Exception:
        conn.rollback()
        raise

    hits = []
    for row_id, text, source, category, subcategory, chunk_index, dist in rows:
        score = 1 / (1 + dist)
        if score >= min_score:
            hits.append({
                "id": row_id,
                "text": text,
                "source": source,
                "score": round(score, 4),
                "category": category,
                "subcategory": subcategory,
                "chunk_index": chunk_index,
            })

    return hits


# ─────────────────────────────────────────────
# CONTEXT FORMATTING
# ─────────────────────────────────────────────

def format_context_block(rag_hits: list[dict]) -> str:
    if not rag_hits:
        return ""
    blocks = []
    for hit in rag_hits:
        blocks.append(
            f"[SOURCE: {hit.get('source', 'unknown')} | "
            f"cat: {hit.get('category', 'unknown')}/{hit.get('subcategory', 'unknown')} | "
            f"chunk: {hit.get('chunk_index', '?')} | "
            f"id: {hit.get('id', '?')} | "
            f"score: {hit.get('score', '?')}]\n"
            f"{hit.get('text', '')}"
        )
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────
# COUNTS / LISTING
# ─────────────────────────────────────────────

def get_db_count() -> int:
    conn = get_chroma_collection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    return cur.fetchone()[0]


def count_documents() -> int:
    return get_db_count()


def list_categories(collection=None) -> list[str]:
    conn = get_chroma_collection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM documents")
    return [r[0] for r in cur.fetchall()]


def get_by_category(category: str, subcategory: str = None, collection=None) -> list[dict]:
    conn = get_chroma_collection()
    cur = conn.cursor()

    if subcategory:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index
            FROM documents WHERE category = %s AND subcategory = %s ORDER BY id
            """,
            (category, subcategory),
        )
    else:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index
            FROM documents WHERE category = %s ORDER BY id
            """,
            (category,),
        )

    return [
        {"id": r[0], "text": r[1], "source": r[2],
         "category": r[3], "subcategory": r[4], "chunk_index": r[5]}
        for r in cur.fetchall()
    ]
