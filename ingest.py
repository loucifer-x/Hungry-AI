"""
ingest.py — Unified RAG ingestion pipeline (docs + web + crawl) — async edition

python3 ingest.py --mode web --url https://github.com/louuuuuu/Hackopedia/tree/main/Web%20Hacking
python3 ingest.py --mode web --url https://github.com/louuuuuu/Hackopedia --crawl --prefix https://github.com/louuuuuu/Hackopedia --max-pages 100
python3 ingest.py --mode docs --path README.md

pip install aiohttp
"""

from __future__ import annotations
import re
import asyncio
import argparse
import logging
import sys
import time
import unicodedata
import math
from collections import Counter
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
import requests
from bs4 import BeautifulSoup

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
import chromadb
import os
import config
from rag import chunk_text, upsert_chunks, get_chroma_collection, list_categories, get_by_category

import trafilatura
from readability import Document
import shutil

config.configure_logging()
logger = logging.getLogger(__name__)
console = Console()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

# GitHub UI paths that have no useful content — skip them during crawl
GITHUB_SKIP_PATHS = {
    "/network", "/pulse", "/graphs", "/settings", "/issues",
    "/pulls", "/actions", "/projects", "/security", "/insights",
    "/stargazers", "/watchers", "/forks", "/commits", "/branches",
    "/tags", "/releases", "/packages", "/discussions", "/wiki",
    "/compare", "/blame", "/raw", "/edit", "/delete", "/find",
}

# Nav/boilerplate lines to strip from extracted text
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(skip to|jump to|back to|go to) (content|main|top|navigation)",
        r"^(home|about|contact|privacy|terms|cookies?|sign (in|up)|log (in|out)|register)",
        r"^(loading\.\.\.|please wait|javascript is required)",
        r"^(copyright|all rights reserved|©)",
        r"^\s*[\|\/\\]\s*$",
        r"^(tweet|share|like|follow|subscribe)",
        r"^(menu|sidebar|footer|header|navigation|breadcrumb)",
        r"^\d+\s*(views?|comments?|shares?|likes?)\s*$",
        r"^(read more|see more|show more|expand|collapse)\s*\.?$",
    ]
]

_GIBBERISH_PATTERNS = [
    re.compile(p) for p in [
        r"^[^a-zA-Z0-9\s]{5,}$",
        r"(.)\1{6,}",
        r"^[a-f0-9]{32,}$",
        r"^\s*\[[\w\s]{0,12}\]\s*$",
    ]
]


# ─────────────────────────────────────────────
# SOURCE SIGNAL BUILDER
# ─────────────────────────────────────────────

def build_source_signal(url_or_path: str, content: str = "") -> str:
    """
    Produce a clean, keyword-rich string that infer_category() can score
    reliably, regardless of whether the input is a raw GitHub URL, a local
    file path, or a generic web URL.

    Strategy (applied in order):
      1. Percent-decode the URL so %20 becomes a space, etc.
      2. For GitHub blob/tree URLs strip the noisy user/repo/branch prefix and
         keep only the meaningful path (folder + filename).
      3. For local paths keep parent-directory name + filename.
      4. Replace path separators and punctuation with spaces.
      5. Extract the page/file title from the first heading or first line of
         real content and append it — this gives the categoriser actual topic
         words rather than just filename tokens.

    The result is lowercased and returned.  It is passed to upsert_chunks()
    as the ``source`` argument so that the DB record also carries readable
    provenance.
    """
    raw = unquote(url_or_path)

    if "github.com" in raw:
        # Strip everything up to and including /blob/<branch>/ or /tree/<branch>/
        m = re.search(r"/(?:blob|tree)/[^/]+/(.+)$", raw)
        if m:
            path_part = m.group(1)          # e.g. "Web Hacking/SQL Injection.md"
        else:
            parts = [p for p in urlparse(raw).path.split("/") if p]
            path_part = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else raw)
    else:
        p = Path(raw)
        if p.parent.name not in (".", ""):
            path_part = f"{p.parent.name}/{p.name}"
        else:
            path_part = p.name

    # Replace separators with spaces for keyword matching
    signal = re.sub(r"[/_\-%.]+", " ", path_part).strip()

    # Extract a content hint: first H1/H2 heading or first non-empty line
    content_hint = _extract_title_hint(content)
    if content_hint:
        signal = f"{signal} {content_hint}"

    return signal.lower()


def _extract_title_hint(text: str, max_chars: int = 120) -> str:
    """
    Pull the first meaningful heading or sentence from extracted text.
    We cap it so we don't overwhelm the keyword scorer with body prose.
    """
    if not text:
        return ""

    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        # Skip very short lines, pure-symbol lines, and lines that look like
        # UI chrome (single words, navigation items, etc.)
        if len(stripped) < 8:
            continue
        if not re.search(r"[a-zA-Z]{3,}", stripped):
            continue
        return stripped[:max_chars]

    return ""


# ─────────────────────────────────────────────
# NOISE FILTER
# ─────────────────────────────────────────────

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def dedupe_chunks(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        norm = " ".join(c.lower().split())
        if norm not in seen:
            seen.add(norm)
            out.append(c)
    return out


def is_bad_chunk(text: str) -> bool:
    words = text.split()
    if len(words) < 5:
        return True
    if len(set(words)) / max(len(words), 1) < 0.3:
        return True
    return False


def _is_gibberish_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _GIBBERISH_PATTERNS:
        if pat.search(s):
            return True
    if len(s) > 20 and _entropy(s) < 1.5:
        return True
    non_print = sum(1 for c in s if unicodedata.category(c) in ("Cc", "Cf", "Cn"))
    if len(s) > 5 and non_print / len(s) > 0.25:
        return True
    return False


def _is_boilerplate_line(line: str) -> bool:
    s = line.strip()
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(s):
            return True
    return False


def _word_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    real = sum(1 for t in tokens if re.search(r"[a-zA-Z]{2,}", t))
    return real / len(tokens)


def clean_extracted_text(raw: str) -> str:
    if not raw:
        return ""

    lines = raw.splitlines()
    kept: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _is_boilerplate_line(stripped):
            continue
        if _is_gibberish_line(stripped):
            continue
        if len(stripped) < 4:
            continue
        if len(stripped) > 40 and _word_ratio(stripped) < 0.35:
            continue
        kept.append(line)

    result_lines: list[str] = []
    blank_run = 0
    for line in kept:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                result_lines.append(line)
        else:
            blank_run = 0
            result_lines.append(line)

    cleaned = "\n".join(result_lines).strip()

    if cleaned and _word_ratio(cleaned) < 0.30:
        return ""

    return cleaned


# ─────────────────────────────────────────────
# DOC LOADERS
# ─────────────────────────────────────────────

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path)) or ""
    except ImportError:
        console.print(f"[yellow]⚠ pdfminer not installed: skipping {path.name}[/]")
        return ""
    except Exception as exc:
        console.print(f"[red]✗ PDF error {path.name}: {exc}[/]")
        return ""


LOADER_MAP = {
    ".txt":      load_text_file,
    ".md":       load_text_file,
    ".markdown": load_text_file,
    ".pdf":      load_pdf_file,
}


def load_file(path: Path) -> str | None:
    loader = LOADER_MAP.get(path.suffix.lower())
    return loader(path) if loader else None


def collect_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in LOADER_MAP else []
    return [p for p in root.rglob("*") if p.suffix.lower() in LOADER_MAP]


# ─────────────────────────────────────────────
# GITHUB HELPERS
# ─────────────────────────────────────────────

def is_github_blob(url: str) -> bool:
    return "github.com" in url and "/blob/" in url


def is_github_tree(url: str) -> bool:
    return "github.com" in url and "/tree/" in url


def is_github_junk(url: str) -> bool:
    if "github.com" not in url:
        return False
    parsed = urlparse(url)
    path = parsed.path
    for skip in GITHUB_SKIP_PATHS:
        if path.endswith(skip) or f"{skip}/" in path:
            return True
    return bool(parsed.query)


def to_github_raw(url: str) -> str:
    parts = url.split("github.com/")[1].split("/")
    user, repo = parts[0], parts[1]
    branch_i = parts.index("blob") + 1
    branch = parts[branch_i]
    path = "/".join(parts[branch_i + 1:])
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"


def extract_github_links(html: str, base_url: str, allowed_prefix: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#"):
            continue
        full = urljoin(base_url, href).split("#")[0]
        if not full.startswith(allowed_prefix):
            continue
        if full == base_url:
            continue
        if is_github_junk(full):
            continue
        parsed = urlparse(full)
        path = parsed.path
        if "/tree/" in path or "/blob/" in path:
            found.add(full)
    return list(found)


def extract_github_file_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if "/blob/" in href:
            full = urljoin("https://github.com", href).split("#")[0]
            found.add(full)
    return list(found)


# ─────────────────────────────────────────────
# TEXT EXTRACTORS
# ─────────────────────────────────────────────

def extract_trafilatura(html: str) -> str | None:
    return trafilatura.extract(html)


def extract_readability(html: str) -> str:
    doc = Document(html)
    soup = BeautifulSoup(doc.summary(), "html.parser")
    return soup.get_text("\n")


def extract_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    if not main:
        return ""
    return "\n".join(main.stripped_strings)


def extract_links(html: str, base_url: str, allowed_prefix: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, tag["href"].strip()).split("#")[0]
        if full.startswith(allowed_prefix) and full != base_url and not is_github_junk(full):
            found.add(full)
    return list(found)


def parse_html(html: str, url: str, content_type: str) -> str:
    if "text/plain" in content_type or url.endswith(".md"):
        raw = html
    else:
        raw = extract_trafilatura(html)
        if not raw or len(raw) < 200:
            try:
                raw = extract_readability(html)
            except Exception:
                raw = None
        if not raw or len(raw) < 200:
            raw = extract_bs4(html)
    return clean_extracted_text(raw or "")


# ─────────────────────────────────────────────
# ASYNC FETCH
# ─────────────────────────────────────────────

async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    *,
    semaphore: asyncio.Semaphore,
    delay: float = 0.5,
) -> tuple[str, str, str]:
    fetch_url = to_github_raw(url) if is_github_blob(url) else url
    async with semaphore:
        await asyncio.sleep(delay)
        try:
            async with session.get(fetch_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as r:
                r.raise_for_status()
                html = await r.text(errors="replace")
                content_type = r.headers.get("Content-Type", "")
                return url, html, content_type
        except Exception as e:
            raise RuntimeError(f"Fetch failed: {e}")


# ─────────────────────────────────────────────
# ASYNC CRAWLER
# ─────────────────────────────────────────────

async def crawl_urls_async(
    seed_urls: list[str],
    *,
    allowed_prefix: str | None = None,
    max_pages: int = 50,
    concurrency: int = 8,
    delay: float = 0.5,
) -> list[str]:
    if allowed_prefix is None:
        parsed = urlparse(seed_urls[0])
        allowed_prefix = f"{parsed.scheme}://{parsed.netloc}"

    is_github = "github.com" in allowed_prefix

    console.print(f"[dim]Crawl prefix : {allowed_prefix}[/]")
    console.print(f"[dim]Concurrency  : {concurrency}[/]")
    if is_github:
        console.print(f"[dim]GitHub mode  : folder traversal enabled[/]")

    visited: set[str] = set()
    queue: list[str] = [u for u in seed_urls if not is_github_junk(u)]
    content_urls: set[str] = set()

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        while queue and len(visited) < max_pages * 3:
            batch: list[str] = []
            while queue and len(batch) < concurrency:
                url = queue.pop(0)
                if url not in visited:
                    batch.append(url)
                    visited.add(url)

            if not batch:
                break

            tasks = [fetch(session, url, semaphore=semaphore, delay=delay) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    console.print(f"[yellow]⚠ Crawl skip: {result}[/]")
                    continue

                url, html, content_type = result

                if is_github:
                    if is_github_blob(url):
                        content_urls.add(url)
                    elif is_github_tree(url) and "text/html" in content_type:
                        new_links = extract_github_links(html, url, allowed_prefix)
                        for link in new_links:
                            if link not in visited:
                                queue.append(link)
                        file_links = extract_github_file_links(html, url)
                        for fl in file_links:
                            if fl not in visited and fl.startswith(allowed_prefix):
                                content_urls.add(fl)
                    elif "text/html" in content_type:
                        new_links = extract_github_links(html, url, allowed_prefix)
                        for link in new_links:
                            if link not in visited:
                                queue.append(link)
                else:
                    content_urls.add(url)
                    if "text/html" in content_type:
                        new_links = extract_links(html, url, allowed_prefix)
                        for link in new_links:
                            if link not in visited and len(visited) + len(queue) < max_pages * 4:
                                queue.append(link)

                if len(content_urls) >= max_pages:
                    queue.clear()
                    break

    result_urls = list(content_urls)[:max_pages]
    console.print(f"[dim]Crawl found {len(result_urls)} content pages (visited {len(visited)} total)[/]")
    return result_urls


# ─────────────────────────────────────────────
# ASYNC WEB PAGE LOADER
# ─────────────────────────────────────────────

async def load_web_page_async(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    delay: float = 0.5,
) -> str:
    url_result, html, content_type = await fetch(session, url, semaphore=semaphore, delay=delay)
    return parse_html(html, url_result, content_type)


# ─────────────────────────────────────────────
# DB HANDLE
# ─────────────────────────────────────────────

col = get_chroma_collection()


# ─────────────────────────────────────────────
# INGEST DOCS
# ─────────────────────────────────────────────

def ingest_docs(path: Path):
    files = collect_files(path)
    if not files:
        console.print("[yellow]No valid files found[/]")
        return 0, 0, 0, []

    total_chunks = 0
    skipped = 0
    scraped_data: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting docs...", total=len(files))

        for f in files:
            progress.update(task, description=f"[red]{f.name}[/]")
            text = load_file(f)
            if not text or not text.strip():
                skipped += 1
                progress.advance(task)
                continue

            text = clean_extracted_text(text)
            if not text:
                skipped += 1
                progress.advance(task)
                continue

            chunks = chunk_text(text)
            chunks = dedupe_chunks(chunks)
            chunks = [c for c in chunks if not is_bad_chunk(c)]

            # Build a signal that combines the file path with a content hint
            # so categorisation uses topic words, not just the filename.
            source_signal = build_source_signal(str(f), text)

            n = upsert_chunks(chunks, source=source_signal, collection=col)
            total_chunks += n
            scraped_data.append(f"[DOC] {f.name}\n{text}\n")
            progress.advance(task)

    return len(files), total_chunks, skipped, scraped_data


# ─────────────────────────────────────────────
# INGEST WEB
# ─────────────────────────────────────────────

async def ingest_web_async(
    urls: list[str],
    *,
    crawl: bool = False,
    allowed_prefix: str | None = None,
    max_pages: int = 50,
    concurrency: int = 8,
    delay: float = 0.5,
):
    total_chunks = 0
    scraped_data: list[str] = []

    if crawl:
        console.print("[bold red]Crawl mode enabled — discovering links...[/]")
        urls = await crawl_urls_async(
            urls,
            allowed_prefix=allowed_prefix,
            max_pages=max_pages,
            concurrency=concurrency,
            delay=delay,
        )

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)

    noise_skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting web...", total=len(urls))

        async with aiohttp.ClientSession(connector=connector) as session:

            async def process(url: str):
                nonlocal total_chunks, noise_skipped
                progress.update(task, description=f"[red]{url[:60]}[/]")
                try:
                    text = await load_web_page_async(session, url, semaphore, delay)
                except Exception as e:
                    console.print(f"[red]✗ Failed {url}: {e}[/]")
                    progress.advance(task)
                    return

                if not text.strip():
                    noise_skipped += 1
                    progress.advance(task)
                    return

                word_count = len(text.split())
                if word_count < 30:
                    console.print(f"[dim]⚠ Skipped (too short, {word_count}w): {url[:80]}[/]")
                    noise_skipped += 1
                    progress.advance(task)
                    return

                chunks = chunk_text(text)
                chunks = dedupe_chunks(chunks)
                chunks = [c for c in chunks if not is_bad_chunk(c)]

                # Build a signal that combines the decoded URL path with the
                # page's own title/heading — this is the key fix: raw GitHub
                # URLs like /louuuuuu/Hackopedia/blob/main/... contain no
                # useful topic words, but the decoded path + first heading do.
                source_signal = build_source_signal(url, text)

                n = upsert_chunks(chunks, source=source_signal, collection=col)
                total_chunks += n
                scraped_data.append(f"[WEB] {url}\n{text}\n")
                progress.advance(task)

            await asyncio.gather(*[process(u) for u in urls])

    if noise_skipped:
        console.print(f"[dim]Noise/stub pages skipped: {noise_skipped}[/]")

    return len(urls), total_chunks, noise_skipped, scraped_data


def ingest_web(urls, *, crawl=False, allowed_prefix=None, max_pages=50, concurrency=8, delay=0.5):
    """Sync entry point."""
    return asyncio.run(
        ingest_web_async(
            urls,
            crawl=crawl,
            allowed_prefix=allowed_prefix,
            max_pages=max_pages,
            concurrency=concurrency,
            delay=delay,
        )
    )
# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["docs", "web"], required=False)
    parser.add_argument("--path", default=str(config.DOCS_DIR))
    parser.add_argument("--url", action="append")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--removeall", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--crawl", action="store_true", help="Auto-discover links from seed URLs")
    parser.add_argument("--prefix", default=None, help="Constrain crawl to URLs starting with this prefix")
    parser.add_argument("--max-pages", type=int, default=50, help="Max content pages to ingest (default: 50)")
    parser.add_argument("--concurrency", type=int, default=8, help="Async concurrency (default: 8)")
    parser.add_argument("--delay", type=float, default=0.5, help="Per-request delay in seconds (default: 0.5)")
    parser.add_argument("--help", action="store_true")

    args = parser.parse_args()

    db_path = "chroma_db"
    if args.removeall:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print("chroma_db deleted.")
        else:
            print("chroma_db does not exist.")
        exit()

    console.rule("[bold red]RAG Ingestion Pipeline[/]")

    if args.remove:
        col = get_chroma_collection()
        section = args.remove
        try:
            result = col.get(where={"source": section}, include=["ids"])
            ids = result.get("ids", [])
            if not ids:
                print(f"⚠ No vectors found for section: {section}")
                exit()
            col.delete(ids=ids)
            print(f"🗑 Deleted section: {section}")
            print(f"🧠 Removed chunks: {len(ids)}")
            print(f"📦 Remaining vectors: {col.count()}")
        except Exception as e:
            print(f"❌ Failed to delete section: {e}")
        exit()

    if args.check:
        repl()

    if args.help:
        console.print(
            Panel.fit(
                "[bold red]COMMANDS[/]\n\n"
                "[red]--mode web --url <url>[/]                            → ingest web pages\n"
                "[red]--mode web --url <url> --crawl[/]                    → crawl + ingest\n"
                "[red]--mode web --url <url> --crawl --prefix <pfx>[/]     → crawl with prefix constraint\n"
                "[red]--mode web --url <url> --crawl --max-pages 100[/]    → crawl up to 100 pages\n"
                "[red]--concurrency 16[/]                                  → parallel requests (default 8)\n"
                "[red]--delay 0.3[/]                                       → seconds between requests (default 0.5)\n"
                "[red]--mode doc --path <path>[/]                          → ingest local files\n"
                "[red]--check[/]                                           → check storage\n"
                "[red]--help[/]                                            → show help\n"
                "[bold red]--remove[/]                                     → deletes ALL data\n",
                border_style="red",
                padding=(1, 2),
            )
        )

    if args.reset:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        client.delete_collection(config.CHROMA_COLLECTION)
        col = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        console.print("[yellow]🗑 Reset complete[/]")

    console.print(f"Store: [red]{config.CHROMA_DIR}[/]")

    scraped_data = []

    if args.mode == "docs":
        source = Path(args.path).expanduser().resolve()
        if not source.exists():
            console.print("[red]Path not found[/]")
            sys.exit(1)
        console.print(f"Mode : docs\nSource: {source}\n")
        files, chunks, skipped, scraped_data = ingest_docs(source)

    elif args.mode == "web":
        if not args.url:
            console.print("[red]No URLs provided[/]")
            sys.exit(1)
        console.print("Mode : web\n")
        files, chunks, skipped, scraped_data = ingest_web(
            args.url,
            crawl=args.crawl,
            allowed_prefix=args.prefix,
            max_pages=args.max_pages,
            concurrency=args.concurrency,
            delay=args.delay,
        )

    else:
        sys.exit(1)

    bannerr = r"""
    ██████ ██████   █████  ██     ██ ██      ███████ ██████  
    ██      ██   ██ ██   ██ ██     ██ ██      ██      ██   ██ 
    ██      ██████  ███████ ██  █  ██ ██      █████   ██   ██ 
    ██      ██   ██ ██   ██ ██ ███ ██ ██      ██      ██   ██ 
    ██████ ██   ██ ██   ██  ███ ███  ███████ ███████ ██████  
    """
    print(bannerr)
    col = get_chroma_collection()
    console.print(f"[red]Sources:[/] {files}")
    console.print(f"[red]Chunks:[/] {chunks}")
    console.print(f"[red]Skipped/noise:[/] {skipped}")
    console.rule("[bold red]Ingestion Summary[/]")

    console.rule("[bold red]Scraped Content Preview[/]")
    for it in scraped_data:
        cleaned = re.sub(r"\s+", " ", it).strip()
        chunks_list = chunk_text(cleaned)
        source_line = it.split("\n")[0]
        console.print(f"[bold red]source:[/] {source_line}")
        console.print(f"[red]chunks:[/] {len(chunks_list)}")
        console.print(f"[red]words:[/]  {len(cleaned.split())}")
        console.print(f"[red]chars:[/]  {len(cleaned)}")
        console.print("[dim]" + "-" * 80 + "[/dim]")


# ----------------------------
# REPL
# ----------------------------
def repl():
    conn = get_chroma_collection()
    console.print("\n[bold red]RAG Interactive Mode[/]\nType 'help' for commands.\n")

    while True:
        cmd = input("rag>").strip()
        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action == "help":
            console.print(
                Panel.fit(
                    "[bold red]Commands[/]\n\n"
                    "[red]list[/]                    → list categories\n"
                    "[red]list <category>[/]         → list items\n"
                    "[red]list <category> --a[/]     → aggressive mode (no whitespace)\n"
                    "[red]status[/]                  → show storage info\n"
                    "[red]clear[/]                   → clear screen\n"
                    "[bold bright_red]exit[/]        → quit REPL",
                    title="Help",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        elif action == "exit":
            break

        elif action == "remove":
            TABLE = "documents"
            if len(parts) < 2:
                console.print("[red]Usage: remove <id> OR remove <category> [id]")

            arg1 = parts[1]
            arg2 = parts[2] if len(parts) > 2 else None

            try:
                if arg1.isdigit() and arg2 is None:
                    where_clause = "id = %s"
                    params = [int(arg1)]
                    target = f"id={arg1}"
                elif not arg1.isdigit() and arg2 and arg2.isdigit():
                    where_clause = "category = %s AND id = %s"
                    params = [arg1, int(arg2)]
                    target = f"{arg1} id={arg2}"
                else:
                    category = arg1
                    where_clause = "category = %s"
                    params = [category]
                    target = f"ALL in {category}"

                with conn.cursor() as cur:
                    cur.execute(f"SELECT id FROM {TABLE} WHERE {where_clause}", params)
                    rows = cur.fetchall()

                if not rows:
                    console.print(f"[yellow]No items found for:[/] {target}")
                    continue

                console.print(f"[yellow]Deleting {len(rows)} items: {target}? (y/n)[/]")
                if input("> ").strip().lower() != "y":
                    continue

                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {TABLE} WHERE {where_clause}", params)

                conn.commit()
                console.print(f"[green]🗑 Deleted:[/] {target}")

            except Exception as e:
                conn.rollback()
                console.print(f"[red]Delete failed:[/] {e}")

        elif action == "status":
            console.print(f"Chunks stored: {col.count()}")

        elif action == "list":
            if len(parts) < 2:
                cats = list_categories(col)
                console.rule("[bold red]CATEGORIES[/]")
                for c in cats:
                    items = get_by_category(c, collection=col)
                    subcats = {
                        it.get("subcategory", "none")
                        for it in items
                        if isinstance(it, dict)
                    }
                    chunk_count = len(items)
                    word_count = sum(len(it.get("text", "").split()) for it in items)
                    console.print(
                        f"[bold red] - {c} [/]"
                        f"[white](chunks={chunk_count} | words={word_count}) "
                        f"| sub category: {', '.join(subcats)}[/]"
                    )
                continue

            category = parts[1]
            subcategory = parts[2] if len(parts) > 2 and not parts[2].startswith("-") else None
            items = get_by_category(category, collection=col)

            if subcategory:
                items = [it for it in items if it.get("subcategory", "none") == subcategory]

            if not items:
                console.print("[blue]No items found in category[/]")
                continue

            console.rule(
                f"[bold red]Items in category: {category}"
                + (f" | subcategory: {subcategory}" if subcategory else "")
            )

            subcats = sorted({
                it.get("subcategory", "none")
                for it in items
                if isinstance(it, dict)
            })

            for it in items:
                db_id = it.get("id")
                source = it.get("source", "?")
                text = it.get("text", "")
                chunk_index = it.get("chunk_index", "?")
                header = (
                    f"[bold red]ID [{db_id}] "
                    f"| chunk {chunk_index} "
                    f"| sub categories = {', '.join(subcats)} "
                    f"| {source}[/]"
                )
                console.rule(header)
                console.rule("")
                if "--a" in parts or "-a" in parts:
                    text = re.sub(r"\s+", "", text)
                console.print(text, markup=False)
                console.rule("")

        elif action == "clear":
            try:
                if len(parts) < 2:
                    console.clear()
                    continue
                items = get_by_category(parts[1], col)
                for i, it in enumerate(items[:20]):
                    console.print(f"\n[{i}] {it['source']}")
                    console.print(it["text"][:400])
            except Exception:
                pass

        else:
            console.print("Unknown command")


if __name__ == "__main__":
    main()
    banner = r"""
    ██████ ██████   █████  ██     ██ ██      ███████ ██████  
    ██      ██   ██ ██   ██ ██     ██ ██      ██      ██   ██ 
    ██      ██████  ███████ ██  █  ██ ██      █████   ██   ██ 
    ██      ██   ██ ██   ██ ██ ███ ██ ██      ██      ██   ██ 
    ██████ ██   ██ ██   ██  ███ ███  ███████ ███████ ██████  
    """
