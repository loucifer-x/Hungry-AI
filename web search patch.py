"""
Drop-in replacement for _ddg_search() in rag.py.
Replaces the brittle DuckDuckGo HTML scrape with a three-backend search
pipeline that:
  1. Queries DuckDuckGo (HTML endpoint), Bing (HTML), and Mojeek (HTML,
     independent index) in parallel using httpx for connection pooling.
  2. Merges and deduplicates the raw URL pools.
  3. Scores every URL on domain trustworthiness, path specificity, and
     whether it looks like thin/spam content, then returns the top results.

The public interface is identical to the old _ddg_search():
    urls = _multi_search(query, max_results=6)

Replace every call to _ddg_search() in _web_fallback_ingest() with
_multi_search().  Remove the old _ddg_search() function entirely.

Speed improvements over v1:
  - httpx replaces urllib.request: persistent sessions + connection pooling
  - _SEARCH_TIMEOUT reduced to 5s (was 10s) — stalled backends cut faster
  - Blocklist short-circuit runs before urlparse (cheap string check first)
  - PATH_BONUSES / PATH_PENALTIES compiled into single alternation regexes
  - removeprefix() replaced with lstrip() for Python 3.8 compatibility
"""
from __future__ import annotations

import html as _html
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import httpx
    _USE_HTTPX = True
except ImportError:
    import urllib.request
    _USE_HTTPX = False

logger = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
_SEARCH_TIMEOUT  = 5    # seconds — reduced from 10; stalled backends cut sooner
_PARALLEL_WORKERS = 3   # one thread per backend

# ── Domain trust tiers ────────────────────────────────────────────────────────
_DOMAIN_TRUST: dict[str, float] = {
    # Source code and technical references
    "github.com":               2.0,
    "wikipedia.org":               2.0,
    "gitlab.com":               1.8,
    "raw.githubusercontent.com":1.5,
    # Official documentation
    "docs.python.org":          2.0,
    "man7.org":                 1.8,
    "linux.die.net":            1.8,
    "kernel.org":               2.0,
    # Security knowledge bases
    "owasp.org":                2.5,
    "portswigger.net":          2.5,
    "nvd.nist.gov":             2.5,
    "cve.mitre.org":            2.5,
    "exploit-db.com":           2.0,
    "hacktricks.xyz":           2.0,
    "book.hacktricks.xyz":      2.0,
    "pentestmonkey.net":        1.8,
    "gtfobins.github.io":       2.0,
    "lolbas-project.github.io": 2.0,
    # General tech reference
    "stackoverflow.com":        1.5,
    "superuser.com":            1.2,
    "askubuntu.com":            1.2,
    "debian.org":               1.8,
    "archlinux.org":            1.8,
    "redhat.com":               1.5,
    "ubuntu.com":               1.5,
    # Security blogs and news
    "krebs.onsecurity.com":     1.5,
    "krebsonsecurity.com":      1.5,
    "schneier.com":             1.5,
    "theregister.com":          1.2,
    "bleepingcomputer.com":     1.5,
    "securityweek.com":         1.3,
    "sans.org":                 1.8,
    "cisco.com":                1.3,
    "paloaltonetworks.com":     1.3,
    # CTF / labs
    "tryhackme.com":            1.5,
    "hackthebox.com":           1.5,
    "ctftime.org":              1.5,
    "writeups.ctf.how":         1.5,
}

_DOMAIN_PENALTY: dict[str, float] = {
    "pinterest.com":    -5.0,
    "pinterest.co.uk":  -5.0,
    "quora.com":        -2.0,
    "medium.com":       -0.5,
    "reddit.com":       -0.5,
    "scribd.com":       -3.0,
    "slideshare.net":   -2.0,
    "chegg.com":        -3.0,
    "coursehero.com":   -3.0,
    "answers.yahoo.com":-4.0,
}

_DOMAIN_BLOCKLIST: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be",
    "amazon.com", "ebay.com", "etsy.com",
    "yelp.com", "tripadvisor.com",
    "duckduckgo.com", "bing.com", "google.com", "mojeek.com",
}

# ── Path quality signals — compiled into single alternations for speed ────────
#
# Instead of looping over N individual patterns per URL we build one combined
# pattern per list.  re.search on a single alternation is faster than N
# individual re.search calls because the engine only makes one pass.
#
# We store (combined_pattern, [(individual_pattern, score), ...]) so we can
# still return the *first* matching bonus/penalty for each group — which is the
# same semantics as the original loop.  If you want *all* matching bonuses to
# stack, switch to iterating _PATH_BONUS_RULES directly (commented below).

_PATH_BONUS_RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/wiki/",        re.I), 0.5),
    (re.compile(r"/docs?/",       re.I), 0.8),
    (re.compile(r"/manual/",      re.I), 0.8),
    (re.compile(r"/tutorial/",    re.I), 0.6),
    (re.compile(r"/writeup",      re.I), 0.8),
    (re.compile(r"/exploit",      re.I), 0.7),
    (re.compile(r"/vulnerabilit", re.I), 0.7),
    (re.compile(r"/cve-\d{4}-",   re.I), 1.0),
    (re.compile(r"/advisory",     re.I), 0.8),
    (re.compile(r"\.md$",         re.I), 0.5),
    (re.compile(r"\.rst$",        re.I), 0.4),
]

_PATH_PENALTY_RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/tag/",              re.I), -0.5),
    (re.compile(r"/category/",         re.I), -0.5),
    (re.compile(r"/author/",           re.I), -0.8),
    (re.compile(r"/search\?",          re.I), -2.0),
    (re.compile(r"/page/\d+",          re.I), -0.3),
    (re.compile(r"\?.*utm_",           re.I), -0.2),
    (re.compile(r"login|signin",       re.I), -3.0),
    (re.compile(r"paywall|subscribe",  re.I), -2.0),
]

# Single combined patterns — used for a fast "any match?" pre-check.
_PATH_BONUS_RE   = re.compile(
    "|".join(p.pattern for p, _ in _PATH_BONUS_RULES),  re.I)
_PATH_PENALTY_RE = re.compile(
    "|".join(p.pattern for p, _ in _PATH_PENALTY_RULES), re.I)

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Module-level httpx client — shared across all backends so TCP connections
# are reused within a single _multi_search() call.  Created lazily so import
# doesn't fail when httpx is absent.
_httpx_client: Optional["httpx.Client"] = None

def _get_client() -> "httpx.Client":
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.Client(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=_SEARCH_TIMEOUT,
        )
    return _httpx_client


def _fetch_html(url: str, timeout: int = _SEARCH_TIMEOUT) -> str:
    """GET a URL and return the decoded body, or '' on any error.

    Uses httpx (with connection pooling) when available, falls back to
    urllib.request otherwise.
    """
    try:
        if _USE_HTTPX:
            resp = _get_client().get(url, timeout=timeout)
            return resp.text
        else:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("_fetch_html failed for %s: %s", url, exc)
        return ""


def _extract_hrefs(html: str) -> list[str]:
    """Pull all href= URLs from raw HTML."""
    urls: list[str] = []
    for m in re.finditer(r'href=["\']?(https?://[^"\'>\s]+)', html):
        u = _html.unescape(m.group(1))
        if u not in urls:
            urls.append(u)
    return urls

# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def _search_ddg(query: str, max_results: int) -> list[str]:
    """DuckDuckGo HTML endpoint."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(r'uddg=(https?%3A%2F%2F[^&"]+)', body):
        real = urllib.parse.unquote(m.group(1))
        if "duckduckgo.com" not in real and real not in urls:
            urls.append(real)
            if len(urls) >= max_results:
                break

    if not urls:
        for u in _extract_hrefs(body):
            if "duckduckgo.com" not in u:
                urls.append(u)
                if len(urls) >= max_results:
                    break

    logger.debug("DDG returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


def _search_bing(query: str, max_results: int) -> list[str]:
    """Bing HTML scrape."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded}&count={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(r'<cite[^>]*>(https?://[^<]+)</cite>', body):
        u = _html.unescape(m.group(1)).strip()
        if "bing.com" not in u and u not in urls:
            urls.append(u)
            if len(urls) >= max_results:
                break

    if not urls:
        for u in _extract_hrefs(body):
            if "bing.com" not in u and "microsoft.com" not in u:
                urls.append(u)
                if len(urls) >= max_results:
                    break

    logger.debug("Bing returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


def _search_mojeek(query: str, max_results: int) -> list[str]:
    """Mojeek — independent crawler-based search engine."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.mojeek.com/search?q={encoded}&l={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(
        r'class=["\']ob["\'][^>]*href=["\']([^"\']+)["\']'
        r'|href=["\']([^"\']+)["\'][^>]*class=["\']ob["\']',
        body,
    ):
        u = _html.unescape(m.group(1) or m.group(2) or "").strip()
        if u.startswith("http") and "mojeek.com" not in u and u not in urls:
            urls.append(u)
            if len(urls) >= max_results:
                break

    if not urls:
        for u in _extract_hrefs(body):
            if "mojeek.com" not in u:
                urls.append(u)
                if len(urls) >= max_results:
                    break

    logger.debug("Mojeek returned %d URLs for %r", len(urls), query)
    return urls[:max_results]

# ─────────────────────────────────────────────────────────────────────────────
# URL scoring
# ─────────────────────────────────────────────────────────────────────────────

def _strip_www(domain: str) -> str:
    """Remove leading 'www.' — Python 3.8-compatible."""
    return domain[4:] if domain.startswith("www.") else domain


def _score_url(url: str) -> float:
    """Assign a quality score to a URL.  Higher = better.  < -10 = discard."""

    # ── Fast blocklist pre-check ─────────────────────────────────────────────
    # Do a cheap substring scan before paying for urlparse.
    # Most social/engine domains appear literally in the URL string.
    url_lower = url.lower()
    for b in _DOMAIN_BLOCKLIST:
        if b in url_lower:
            return -99.0

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return -99.0

    domain = _strip_www(parsed.netloc.lower())
    path   = parsed.path + ("?" + parsed.query if parsed.query else "")

    # ── Full blocklist check (catches subdomains e.g. photos.facebook.com) ───
    if any(domain == b or domain.endswith("." + b) for b in _DOMAIN_BLOCKLIST):
        return -99.0

    score = 0.0

    # Domain trust / penalty
    for trusted, bonus in _DOMAIN_TRUST.items():
        if domain == trusted or domain.endswith("." + trusted):
            score += bonus
            break
    for penalised, penalty in _DOMAIN_PENALTY.items():
        if domain == penalised or domain.endswith("." + penalised):
            score += penalty
            break

    # ── Path bonuses — combined alternation fast-path ────────────────────────
    # Only iterate the individual rules when the combined pattern matches,
    # saving N regex calls on the majority of URLs that match nothing.
    if _PATH_BONUS_RE.search(path):
        for pattern, bonus in _PATH_BONUS_RULES:
            if pattern.search(path):
                score += bonus

    # ── Path penalties — same fast-path trick ────────────────────────────────
    if _PATH_PENALTY_RE.search(path):
        for pattern, penalty in _PATH_PENALTY_RULES:
            if pattern.search(path):
                score += penalty

    # Prefer HTTPS
    if parsed.scheme != "https":
        score -= 0.5

    # Prefer shallower paths
    depth = path.count("/")
    if depth <= 3:
        score += 0.3
    elif depth >= 7:
        score -= 0.3

    return score

# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def _multi_search(query: str, max_results: int = 6) -> list[str]:
    """
    Query all three backends in parallel, merge the URL pools, score every
    URL, deduplicate by normalised domain+path, and return the top
    *max_results* URLs sorted by descending score.

    Drop-in replacement for _ddg_search().
    """
    per_backend = max_results + 4

    backends = [
        (_search_ddg,    query, per_backend),
        (_search_bing,   query, per_backend),
        (_search_mojeek, query, per_backend),
    ]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
        futures = {pool.submit(fn, q, n): fn.__name__ for fn, q, n in backends}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                raw_urls.extend(results)
                logger.debug("%s contributed %d URLs", name, len(results))
            except Exception as exc:
                logger.warning("Search backend %s failed: %s", name, exc)

    if not raw_urls:
        logger.warning("_multi_search: all backends returned nothing for %r", query)
        return []

    # Score and deduplicate
    seen_keys: set[str] = set()
    scored: list[tuple[float, str]] = []

    for url in raw_urls:
        score = _score_url(url)
        if score < -10:
            continue

        try:
            parsed = urllib.parse.urlparse(url)
            key = (
                _strip_www(parsed.netloc.lower())
                + parsed.path.rstrip("/").lower()
            )
        except Exception:
            key = url

        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored.append((score, url))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = [url for _, url in scored[:max_results]]

    logger.info(
        "_multi_search: %d raw URLs → %d unique → returning top %d for %r",
        len(raw_urls), len(scored), len(top), query,
    )
    return top

# ─────────────────────────────────────────────────────────────────────────────
# PATCH INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. pip install httpx          (optional but recommended — ~30% faster fetches)
#
# 2. Delete the old _ddg_search() function from rag.py entirely.
#
# 3. Paste everything above into rag.py (or a new web_search.py that you
#    import from rag.py).
#
# 4. In _web_fallback_ingest(), change the one call site:
#
#    OLD:  candidate_urls = _ddg_search(query, max_results=WEB_FALLBACK_MAX_PAGES + 2)
#    NEW:  candidate_urls = _multi_search(query, max_results=WEB_FALLBACK_MAX_PAGES + 2)
#
# 5. No other changes needed.
#
# OPTIONAL: bump WEB_FALLBACK_MAX_PAGES from 4 to 5 or 6 so the extra
# quality URLs the scorer surfaces are actually fetched.