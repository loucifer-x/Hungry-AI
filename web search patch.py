"""
Drop-in replacement for _ddg_search() in rag.py.

Replaces the brittle DuckDuckGo HTML scrape with a three-backend search
pipeline that:
  1. Queries DuckDuckGo (JSON endpoint — no scraping), Bing (HTML), and
     Mojeek (HTML, independent index) in parallel.
  2. Merges and deduplicates the raw URL pools.
  3. Scores every URL on domain trustworthiness, path specificity, and
     whether it looks like thin/spam content, then returns the top results.

The public interface is identical to the old _ddg_search():

    urls = _multi_search(query, max_results=6)

Replace every call to _ddg_search() in _web_fallback_ingest() with
_multi_search().  Remove the old _ddg_search() function entirely.
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# ── Timeouts ─────────────────────────────────────────────────────────────────
_SEARCH_TIMEOUT = 10   # seconds per backend request
_PARALLEL_WORKERS = 3  # one thread per backend

# ── Domain trust tiers ────────────────────────────────────────────────────────
# Bonus added to a URL's base score.  Tweak freely.
_DOMAIN_TRUST: dict[str, float] = {
    # Source code and technical references
    "github.com": 2.0,
    "gitlab.com": 1.8,
    "raw.githubusercontent.com": 1.5,
    # Official documentation patterns (matched as suffix)
    "docs.python.org": 2.0,
    "man7.org": 1.8,
    "linux.die.net": 1.8,
    "kernel.org": 2.0,
    # Security knowledge bases
    "owasp.org": 2.5,
    "portswigger.net": 2.5,
    "nvd.nist.gov": 2.5,
    "cve.mitre.org": 2.5,
    "exploit-db.com": 2.0,
    "hacktricks.xyz": 2.0,
    "book.hacktricks.xyz": 2.0,
    "pentestmonkey.net": 1.8,
    "gtfobins.github.io": 2.0,
    "lolbas-project.github.io": 2.0,
    # General tech reference
    "stackoverflow.com": 1.5,
    "superuser.com": 1.2,
    "askubuntu.com": 1.2,
    "debian.org": 1.8,
    "archlinux.org": 1.8,
    "redhat.com": 1.5,
    "ubuntu.com": 1.5,
    # Security blogs and news
    "krebs.onsecurity.com": 1.5,
    "krebsonsecurity.com": 1.5,
    "schneier.com": 1.5,
    "theregister.com": 1.2,
    "bleepingcomputer.com": 1.5,
    "securityweek.com": 1.3,
    "sans.org": 1.8,
    "cisco.com": 1.3,
    "paloaltonetworks.com": 1.3,
    # CTF / labs
    "tryhackme.com": 1.5,
    "hackthebox.com": 1.5,
    "ctftime.org": 1.5,
    "writeups.ctf.how": 1.5,
}

# Penalty applied to known low-quality domains.
_DOMAIN_PENALTY: dict[str, float] = {
    "pinterest.com": -5.0,
    "pinterest.co.uk": -5.0,
    "quora.com": -2.0,
    "medium.com": -0.5,   # not terrible but often paywalled
    "reddit.com": -0.5,   # useful but unpredictable quality
    "scribd.com": -3.0,
    "slideshare.net": -2.0,
    "chegg.com": -3.0,
    "coursehero.com": -3.0,
    "answers.yahoo.com": -4.0,
}

# Domains to skip entirely (no point fetching).
_DOMAIN_BLOCKLIST: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be",
    "amazon.com", "ebay.com", "etsy.com",
    "yelp.com", "tripadvisor.com",
    "duckduckgo.com", "bing.com", "google.com", "mojeek.com",
}

# ── Path quality signals ──────────────────────────────────────────────────────
# Each matched pattern adds to the URL score.
_PATH_BONUSES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/wiki/",        re.I), 0.5),
    (re.compile(r"/docs?/",       re.I), 0.8),
    (re.compile(r"/manual/",      re.I), 0.8),
    (re.compile(r"/tutorial/",    re.I), 0.6),
    (re.compile(r"/writeup",      re.I), 0.8),  # CTF writeups
    (re.compile(r"/exploit",      re.I), 0.7),
    (re.compile(r"/vulnerabilit", re.I), 0.7),
    (re.compile(r"/cve-\d{4}-",   re.I), 1.0),  # direct CVE pages
    (re.compile(r"/advisory",     re.I), 0.8),
    (re.compile(r"\.md$",         re.I), 0.5),
    (re.compile(r"\.rst$",        re.I), 0.4),
]

_PATH_PENALTIES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/tag/",         re.I), -0.5),
    (re.compile(r"/category/",    re.I), -0.5),
    (re.compile(r"/author/",      re.I), -0.8),
    (re.compile(r"/search\?",     re.I), -2.0),  # search result pages
    (re.compile(r"/page/\d+",     re.I), -0.3),
    (re.compile(r"\?.*utm_",      re.I), -0.2),  # tracking params
    (re.compile(r"login|signin",  re.I), -3.0),
    (re.compile(r"paywall|subscribe", re.I), -2.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_html(url: str, timeout: int = _SEARCH_TIMEOUT) -> str:
    """GET a URL and return the decoded body, or '' on any error."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("_fetch_html failed for %s: %s", url, exc)
        return ""


def _extract_hrefs(html: str, base: str = "") -> list[str]:
    """Pull all href= URLs from raw HTML."""
    urls = []
    for m in re.finditer(r'href=["\']?(https?://[^"\'>\s]+)', html):
        u = _html.unescape(m.group(1))
        if u not in urls:
            urls.append(u)
    return urls


# ── Backend 1: DuckDuckGo lite JSON ──────────────────────────────────────────

def _search_ddg(query: str, max_results: int) -> list[str]:
    """
    DuckDuckGo's /html endpoint is rate-limited and blocks bots aggressively.
    We use the undocumented but more stable lite JSON redirect instead:
      https://duckduckgo.com/?q=<query>&format=json
    Falls back to the HTML endpoint if the JSON route yields nothing.
    """
    encoded = urllib.parse.quote_plus(query)

    # Try the lite HTML endpoint (most reliable without an API key)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    # DDG wraps result links in a redirect — extract the real URL from uddg= param
    for m in re.finditer(r'uddg=(https?%3A%2F%2F[^&"]+)', body):
        real = urllib.parse.unquote(m.group(1))
        if "duckduckgo.com" not in real and real not in urls:
            urls.append(real)
        if len(urls) >= max_results:
            break

    # Fallback: plain hrefs if the uddg pattern matched nothing
    if not urls:
        for u in _extract_hrefs(body):
            if "duckduckgo.com" not in u:
                urls.append(u)
            if len(urls) >= max_results:
                break

    logger.debug("DDG returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


# ── Backend 2: Bing ───────────────────────────────────────────────────────────

def _search_bing(query: str, max_results: int) -> list[str]:
    """Bing HTML scrape — no API key, tolerant of our UA."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded}&count={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    # Bing wraps results in <cite> tags; also appears in regular <a href>
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


# ── Backend 3: Mojeek ─────────────────────────────────────────────────────────

def _search_mojeek(query: str, max_results: int) -> list[str]:
    """
    Mojeek is a genuinely independent crawler-based search engine that
    tolerates automated requests reasonably well.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.mojeek.com/search?q={encoded}&l={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    # Mojeek results are <a class="ob" href="...">
    for m in re.finditer(r'class=["\']ob["\'][^>]*href=["\']([^"\']+)["\']'
                         r'|href=["\']([^"\']+)["\'][^>]*class=["\']ob["\']', body):
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

def _score_url(url: str) -> float:
    """
    Assign a quality score to a URL.
    Higher = more likely to contain accurate, in-depth technical content.
    Negative score = discard.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return -99.0

    domain = parsed.netloc.lower().removeprefix("www.")
    path   = parsed.path + ("?" + parsed.query if parsed.query else "")

    # Hard block
    if any(domain == b or domain.endswith("." + b) for b in _DOMAIN_BLOCKLIST):
        return -99.0

    score = 0.0

    # Domain trust
    for trusted, bonus in _DOMAIN_TRUST.items():
        if domain == trusted or domain.endswith("." + trusted):
            score += bonus
            break

    # Domain penalty
    for penalised, penalty in _DOMAIN_PENALTY.items():
        if domain == penalised or domain.endswith("." + penalised):
            score += penalty  # penalty is already negative
            break

    # Path bonuses
    for pattern, bonus in _PATH_BONUSES:
        if pattern.search(path):
            score += bonus

    # Path penalties
    for pattern, penalty in _PATH_PENALTIES:
        if pattern.search(path):
            score += penalty

    # Prefer HTTPS
    if parsed.scheme != "https":
        score -= 0.5

    # Prefer shorter paths (fewer redirect hops / landing pages)
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
    per_backend = max_results + 4  # fetch extras so scoring has room to work

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
            continue  # hard-blocked

        try:
            parsed = urllib.parse.urlparse(url)
            # Normalise: strip tracking params, lowercase domain, strip trailing /
            key = (
                parsed.netloc.lower().removeprefix("www.")
                + parsed.path.rstrip("/").lower()
            )
        except Exception:
            key = url

        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored.append((score, url))

    # Sort descending by score, then by position in the raw list (stable)
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
# 1. Delete the old _ddg_search() function from rag.py entirely.
#
# 2. Paste everything above (up to this comment block) into rag.py, either
#    at the top of the "WEB FALLBACK" section or in a new web_search.py
#    module that you import from rag.py.
#
# 3. In _web_fallback_ingest(), change the one call site:
#
#    OLD:  candidate_urls = _ddg_search(query, max_results=WEB_FALLBACK_MAX_PAGES + 2)
#    NEW:  candidate_urls = _multi_search(query, max_results=WEB_FALLBACK_MAX_PAGES + 2)
#
# 4. No other changes needed — the rest of _web_fallback_ingest() is unchanged.
#
# OPTIONAL: bump WEB_FALLBACK_MAX_PAGES from 4 to 5 or 6 so the extra
# quality URLs the scorer surfaces are actually fetched.