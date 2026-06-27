"""
rag.py — PostgreSQL + pgvector RAG engine
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
import unicodedata
import math
import urllib.parse
import urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List
from urllib.parse import urljoin, urlparse, unquote
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
import config
import psycopg2
from psycopg2 import pool as pg_pool

logger = logging.getLogger(__name__)
console = Console()
_embedder: Optional["SentenceTransformer"] = None

# ── Connection pool — replaces single global connection ───────────────────────
# A single psycopg2 connection is not thread-safe and reconnects on every
# closed/broken state check.  A SimpleConnectionPool keeps 2-10 live
# connections and hands them out per-call, which matters when retrieve_context
# and _web_fallback_ingest run on background threads.
_db_pool: Optional[pg_pool.SimpleConnectionPool] = None

def _get_pool() -> pg_pool.SimpleConnectionPool:
    global _db_pool
    if _db_pool is None or _db_pool.closed:
        _db_pool = pg_pool.SimpleConnectionPool(
            minconn=2,
            maxconn=10,
            dbname="ragdb",
            user="raguser",
            password="ragpass",
            host="localhost",
            port=5432,
        )
    return _db_pool


def get_chroma_collection():
    """Kept for backward compatibility — returns a pooled PostgreSQL connection."""
    return _get_pool().getconn()


def _return_conn(conn) -> None:
    """Return a connection to the pool.  Call after every DB operation."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


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
    ("linux", "shell", 10, ["bash", "zsh", "fish", "sh", "shell scripting", "terminal"]),
    ("linux", "package management", 10, ["apt", "dpkg", "dnf", "rpm", "pacman", "zypper", "apk", "snap", "flatpak"]),
    ("linux", "system administration", 10, ["systemctl", "systemd", "service", "useradd", "usermod", "passwd", "sudo"]),
    ("linux", "permissions", 10, ["chmod", "chown", "chgrp", "umask", "file permissions", "ownership"]),
    ("linux", "process management", 10, ["ps", "top", "htop", "kill", "pkill", "nice", "renice"]),
    ("linux", "networking", 10, ["ssh", "scp", "rsync", "ping", "ip", "ifconfig", "netstat", "ss"]),

    # ── Red team tools ────────────────────────────────────────────────────────
    ("tools", "nmap", 15, ["nmap", "network mapper", "port scan", "port scanning",
                           "service detection", "version detection", "nse",
                           "nmap scripting engine", "host discovery"]),
    ("tools", "hydra", 10, ["hydra", "thc hydra", "password brute force",
                            "login brute force", "credential attack",
                            "ssh brute force", "ftp brute force"]),
    ("tools", "john_the_ripper", 10, ["john the ripper", "john", "john jumbo",
                                      "password cracking", "hash cracking",
                                      "dictionary attack", "wordlist attack"]),
    ("tools", "hashcat", 10, ["hashcat", "gpu cracking", "hash cracking",
                              "password cracking", "offline cracking",
                              "dictionary attack", "mask attack"]),
    ("tools", "gobuster", 10, ["gobuster", "directory enumeration",
                               "directory brute force", "content discovery",
                               "dns enumeration", "vhost enumeration"]),
    ("tools", "ffuf", 10, ["ffuf", "fuzz faster u fool", "web fuzzing",
                           "directory fuzzing", "content discovery",
                           "parameter fuzzing"]),
    ("tools", "burp_suite", 10, ["burp", "burp suite", "burp proxy",
                                 "repeater", "intruder", "web proxy",
                                 "web application testing"]),
    ("tools", "metasploit", 10, ["metasploit", "msfconsole", "msfvenom",
                                 "meterpreter", "exploit framework",
                                 "payload generation"]),
    ("tools", "sqlmap", 10, ["sqlmap", "sql injection automation",
                             "database enumeration", "sqli exploitation",
                             "sql injection scanner"]),
    ("tools", "nikto", 10, ["nikto", "web vulnerability scanner",
                            "web server scanner", "vulnerability scanning"]),
    ("tools", "wpscan", 10, ["wpscan", "wordpress scanner",
                             "wordpress enumeration", "wordpress security"]),
    ("tools", "amass", 10, ["amass", "subdomain enumeration",
                            "asset discovery", "attack surface mapping",
                            "reconnaissance"]),
    ("tools", "subfinder", 10, ["subfinder", "subdomain enumeration",
                                "asset discovery", "reconnaissance"]),
    ("tools", "enum4linux", 10, ["enum4linux", "smb enumeration",
                                 "windows enumeration", "netbios enumeration"]),
    ("tools", "impacket", 10, ["impacket", "psexec", "wmiexec",
                               "secretsdump", "ntlm relay", "smb execution"]),
    ("tools", "responder", 10, ["responder", "llmnr", "nbns",
                                "name resolution poisoning", "credential capture"]),
    ("tools", "crackmapexec", 10, ["crackmapexec", "cme",
                                   "active directory enumeration",
                                   "smb enumeration", "lateral movement"]),
    ("tools", "netexec", 10, ["netexec", "nxc",
                              "active directory enumeration",
                              "smb enumeration", "lateral movement"]),
    ("tools", "wireshark", 10, ["wireshark", "packet analysis",
                                "packet capture", "network analysis",
                                "pcap analysis"]),
    ("tools", "tcpdump", 10, ["tcpdump", "packet capture",
                              "network sniffing", "pcap"]),
    ("tools", "aircrack_ng", 10, ["aircrack-ng", "aircrack",
                                  "wifi auditing", "wireless security",
                                  "wireless cracking", "wpa cracking"]),

    # ── Red Team / Web Vulns ──────────────────────────────────────────────────
    ("redteam", "reverse_shell", 10, ["reverse shell", "revshell", "bind shell", "shell backconnect",
                                      "tcp reverse connection", "attacker shell", "remote shell access",
                                      "netcat reverse shell", "bash reverse shell", "python reverse shell",
                                      "reverse_shell"]),
    ("redteam", "sql_injection", 10, ["sql injection", "sqli", "union select", "sql payload",
                                      "blind sqli", "error-based sqli", "time-based sqli",
                                      "sql_injection", "sqli exploitation"]),
    ("redteam", "xss", 10, ["xss", "cross-site scripting", "dom xss", "reflected xss",
                             "stored xss", "script injection", "html injection"]),
    ("redteam", "rce", 10, ["rce", "remote code execution", "command injection",
                             "os command", "shell exec", "code exec", "arbitrary command"]),
    ("redteam", "csrf", 9, ["csrf", "cross-site request forgery",
                             "state-changing request", "anti-csrf token"]),
    ("redteam", "ssrf", 10, ["ssrf", "server-side request forgery", "internal endpoint",
                              "metadata endpoint", "cloud ssrf"]),
    ("redteam", "xxe", 9, ["xxe", "xml external entity", "entity injection",
                            "xml injection", "external dtd"]),
    ("redteam", "idor", 9, ["idor", "insecure direct object", "broken access control",
                             "unauthorized object access", "object reference"]),
    ("redteam", "path_traversal", 9, ["path traversal", "directory traversal", "../",
                                       "dot dot slash", "local file inclusion", "lfi", "rfi"]),
    ("redteam", "auth_bypass", 10, ["authentication bypass", "auth bypass", "broken auth",
                                     "jwt attack", "session fixation", "credential stuffing",
                                     "privilege escalation", "privesc", "priv esc"]),
    ("redteam", "open_redirect", 7, ["open redirect", "url redirect", "redirect to external",
                                      "unvalidated redirect"]),
    ("redteam", "clickjacking", 7, ["clickjacking", "ui redressing", "frame injection",
                                     "x-frame-options"]),
    ("redteam", "deserialization", 9, ["insecure deserialization", "java deserialization",
                                        "pickle exploit", "deserialization gadget"]),
    ("redteam", "ssti", 9, ["ssti", "server-side template injection", "jinja2 injection",
                             "twig injection", "template injection"]),
    ("redteam", "race_condition", 8, ["race condition", "toctou", "time of check",
                                       "time of use", "concurrency bug"]),
    ("redteam", "business_logic", 8, ["business logic", "logic flaw", "price manipulation",
                                       "quantity bypass", "workflow abuse"]),
    ("redteam", "cve", 10, ["cve", "common vulnerabilities and exposures", "vulnerability id",
                             "security flaw", "exploit database", "known vulnerability",
                             "security patch", "software vulnerability", "exploit identifier"]),

    # ── Network ───────────────────────────────────────────────────────────────
    ("network", "recon", 7, ["port scan", "host discovery",
                              "service enumeration", "banner grab",
                              "network reconnaissance", "active recon"]),
    ("network", "packet_analysis", 7, ["wireshark", "tcpdump", "packet capture", "pcap",
                                        "traffic analysis", "protocol decode"]),
    ("network", "mitm", 9, ["mitm", "man in the middle", "arp spoofing", "ssl strip",
                             "arp poison", "traffic intercept"]),
    ("network", "dns", 7, ["dns poisoning", "dns hijack", "dns spoofing",
                            "dns tunneling", "domain hijack", "dns rebind"]),
    ("network", "vpn", 6, ["vpn", "wireguard", "openvpn", "ipsec",
                            "tunneling protocol", "split tunnel"]),
    ("network", "firewall", 7, ["firewall bypass", "packet filter", "egress filter",
                                 "ingress filter", "acl", "network policy"]),
    ("network", "ids_evasion", 8, ["ids evasion", "ips bypass", "fragmentation attack",
                                    "evasion technique", "signature bypass"]),

    # ── Malware ───────────────────────────────────────────────────────────────
    ("malware", "ransomware", 10, ["ransomware", "file encryption", "ransom demand",
                                    "decrypt files", "ransom note"]),
    ("malware", "trojan", 9, ["trojan", "remote access trojan", "rat", "backdoor",
                               "persistence mechanism"]),
    ("malware", "keylogger", 9, ["keylogger", "keystroke capture", "input capture",
                                  "keylogging", "keystroke logger"]),
    ("malware", "rootkit", 10, ["rootkit", "kernel module", "ring0", "kernel exploit",
                                 "ring zero", "lkm rootkit"]),
    ("malware", "botnet", 9, ["botnet", "c2", "command and control", "bot herder",
                               "ddos bot", "zombie host"]),
    ("malware", "worm", 9, ["worm", "self-replicating", "lateral movement",
                             "propagation", "network worm"]),
    ("malware", "spyware", 8, ["spyware", "adware", "pup", "stalkerware",
                                "monitoring software"]),

    # ── Code / Languages ──────────────────────────────────────────────────────
    ("code", "python", 7, [".py", "python", "django", "flask", "fastapi",
                            "asyncio", "pydantic"]),
    ("code", "javascript", 7, [".js", "javascript", "node.js", "nodejs", "typescript",
                                "ts", "ecmascript", "commonjs", "esm"]),
    ("code", "rust", 7, [".rs", "rust", "cargo", "borrow checker", "ownership",
                          "unsafe rust", "tokio"]),
    ("code", "go", 7, [".go", "golang", "goroutine", "go routine",
                        "go mod", "go context"]),
    ("code", "java", 7, [".java", "java", "jvm", "spring boot", "maven",
                          "gradle", "jar", "bytecode"]),
    ("code", "csharp", 7, [".cs", "c#", "csharp", ".net", "dotnet",
                            "asp.net", "nuget", "blazor"]),
    ("code", "cpp", 7, [".cpp", "c++", "cmake", "llvm", "stl",
                         "memory management", "pointer", "buffer"]),
    ("code", "shell", 8, ["bash", "shell script", ".sh", "zsh", "fish",
                           "posix shell", "sh script", "heredoc"]),
    ("code", "sql", 7, ["sql", "postgres", "postgresql", "mysql", "sqlite",
                         "tsql", "plpgsql", "stored procedure", "orm"]),
    ("code", "infra_as_code", 7, ["terraform", "pulumi", "cloudformation", "ansible",
                                   "puppet", "chef", "helm", "kubernetes yaml"]),

    # ── AI / ML ───────────────────────────────────────────────────────────────
    ("ai", "llm", 8, ["llm", "transformer", "embedding", "rag",
                       "retrieval augmented", "language model",
                       "gpt", "claude", "gemini"]),
    ("ai", "prompt_injection", 10, ["prompt injection", "jailbreak", "ignore previous",
                                     "system prompt leak", "indirect injection", "prompt hack",
                                     "prompt_injection"]),
    ("ai", "model_extraction", 9, ["model extraction", "model inversion",
                                    "membership inference", "training data leak",
                                    "model stealing"]),
    ("ai", "adversarial", 9, ["adversarial example", "adversarial attack",
                               "evasion attack", "poisoning attack", "trojan model"]),
    ("ai", "fine_tuning", 7, ["fine-tuning", "lora", "qlora", "peft", "sft",
                               "rlhf", "dpo", "instruction tuning"]),
    ("ai", "vector_db", 7, ["vector database", "vector store", "pinecone",
                             "weaviate", "chroma", "qdrant", "faiss", "ann"]),
    ("ai", "ml_pipeline", 7, ["mlflow", "kubeflow", "airflow", "feature store",
                               "data pipeline", "model registry"]),

    # ── Cryptography ──────────────────────────────────────────────────────────
    ("crypto", "asymmetric", 8, ["rsa", "elliptic curve", "ecc", "ecdsa", "ecdh",
                                  "public key", "private key", "pki"]),
    ("crypto", "symmetric", 8, ["aes", "des", "3des", "chacha20", "block cipher",
                                 "stream cipher", "key derivation", "kdf"]),
    ("crypto", "hashing", 8, ["sha256", "sha512", "md5", "bcrypt", "argon2",
                               "scrypt", "pbkdf2", "collision", "hash function"]),
    ("crypto", "tls", 8, ["tls", "ssl", "mtls", "certificate", "x.509",
                           "cipher suite", "pfs", "certificate pinning"]),
    ("crypto", "crypto_attack", 9, ["padding oracle", "cbc attack", "ecb mode",
                                     "length extension", "timing attack",
                                     "nonce reuse", "weak random"]),
    ("crypto", "zero_knowledge", 8, ["zero knowledge", "zk proof", "zkp",
                                      "zk-snark", "zk-stark", "commitment scheme"]),

    # ── Social Engineering ────────────────────────────────────────────────────
    ("social", "phishing", 9, ["phishing", "spear phishing", "whaling", "vishing",
                                "smishing", "credential harvest page", "credential harvesting",
                                "credential harvest", "harvest credentials"]),
    ("social", "osint", 7, ["osint", "open source intelligence", "doxing",
                             "footprinting", "recon-ng", "maltego", "shodan"]),
    ("social", "pretexting", 8, ["pretexting", "social engineering", "impersonation",
                                  "vishing script", "pretext call"]),
    ("social", "physical", 8, ["physical security", "tailgating", "badge cloning",
                                "rfid cloning", "lock picking", "physical pentest"]),

    # ── Cloud ─────────────────────────────────────────────────────────────────
    ("cloud", "aws", 7, ["aws", "s3", "ec2", "iam", "lambda", "cloudtrail",
                          "guardduty", "security group", "vpc"]),
    ("cloud", "azure", 7, ["azure", "microsoft azure", "azure ad", "entra",
                            "arm template", "azure devops", "blob storage"]),
    ("cloud", "gcp", 7, ["gcp", "google cloud", "gke", "cloud run",
                          "bigquery", "iam policy", "service account"]),
    ("cloud", "k8s_security", 9, ["kubernetes", "k8s", "pod security", "rbac", "etcd",
                                   "container escape", "namespace isolation"]),
    ("cloud", "iam", 12, ["iam", "identity access", "privilege escalation cloud",
                          "role assumption", "credential leak", "aws keys",
                          "privilege_escalation", "iam_privilege", "iam privesc"]),

    # ── Mobile ────────────────────────────────────────────────────────────────
    ("mobile", "android", 7, ["android", "apk", "smali", "adb", "intent",
                               "broadcast receiver", "content provider"]),
    ("mobile", "ios", 7, ["ios", "ipa", "jailbreak", "mach-o", "swift",
                           "objc", "objective-c", "codesign", "entitlement"]),
    ("mobile", "mobile_pentest", 8, ["frida", "objection", "burp mobile", "mobile pentest",
                                      "binary analysis", "runtime manipulation"]),

    # ── Data & Privacy ────────────────────────────────────────────────────────
    ("data", "exfiltration", 10, ["data exfiltration", "data leak", "data breach",
                                   "sensitive data", "pii leak", "phi leak"]),
    ("data", "forensics", 7, ["digital forensics", "disk image", "memory forensics",
                               "volatility", "timeline analysis", "artifact recovery"]),
    ("data", "privacy", 7, ["gdpr", "ccpa", "pii", "personal data", "data residency",
                             "right to erasure", "data retention"]),

    ("other", "other", 0, []),
]

# ─────────────────────────────────────────────
# Precomputed keyword indexes
# ─────────────────────────────────────────────

_AMBIGUOUS_TOKENS = {
    "sh", "cat", "rm", "mv", "cp", "ls", "cd", "ps", "ip",
    "ss", "ts", "go", ".go", "rs", "cs", "js",
}

_PHRASE_INDEX: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
_TOKEN_INDEX:  dict[str, list[tuple[str, str, int]]] = defaultdict(list)

for _cat, _sub, _score, _kws in RULES:
    for _kw in _kws:
        if _kw in _AMBIGUOUS_TOKENS:
            continue
        if " " in _kw or len(_kw) > 5:
            _PHRASE_INDEX[_kw].append((_cat, _sub, _score))
        else:
            _TOKEN_INDEX[_kw].append((_cat, _sub, _score))

_KEYWORD_INDEX = {**_PHRASE_INDEX, **_TOKEN_INDEX}

# Compile token boundary patterns once at startup, not per-call
_TOKEN_PATTERN_CACHE: dict[str, re.Pattern] = {
    token: re.compile(
        r"(?<![a-zA-Z0-9])" + re.escape(token) + r"(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )
    for token in _TOKEN_INDEX
}

# Compiled paragraph splitter — used in chunk_text
_PARA_SPLIT_RE = re.compile(r"\n{2,}")


def infer_category(source: str) -> Tuple[str, str]:
    s = source.lower()
    tally: dict[tuple[str, str], float] = defaultdict(float)

    for keyword, entries in _PHRASE_INDEX.items():
        if keyword in s:
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 2.0

    for keyword, entries in _TOKEN_INDEX.items():
        if _TOKEN_PATTERN_CACHE[keyword].search(s):
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 1.0

    if not tally:
        #console.rule("\n[bold red]NEW CATEGORIES ADDED[/]")
        from aicategory import classify_text
        try:
            result = classify_text(s)
            #print("worked")
           #print(s, result)
            x = ", ".join(result)
            a, b = x.split(", ")
            console.print(f"\n[bold red]NEW CATEGORIES ADDED[/bold red]  [red]{s} | {x}[/red]")
            return a, b
        except Exception:
            print("Fall Back to unknown categorization")
            words = re.findall(r"[A-Za-z0-9]+", s)
            if words:
                return words[0].lower(), "general"
            return "unknown", "general"

    best_cat, best_sub = max(
        tally.items(),
        key=lambda kv: (kv[1], len(kv[0][1]))
    )[0]
    return best_cat, best_sub


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    # Use pre-compiled splitter instead of re.split inline
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
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
    try:
        cur = conn.cursor()
        embedder = get_embedder()
        category, subcategory = infer_category(source)

        # Batch-encode all chunks in one call (uses model's internal batching)
        embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False).tolist()

        # executemany instead of one INSERT per chunk — single round-trip
        cur.executemany(
            """
            INSERT INTO documents (source, chunk_index, text, category, subcategory, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (source, i, text, category, subcategory, vec)
                for i, (text, vec) in enumerate(zip(chunks, embeddings))
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _return_conn(conn)

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
    q = query.strip().lower()
    if len(q) < RAG_MIN_LENGTH:
        return False
    if q in RAG_SKIP_PHRASES:
        return False
    for keyword in _KEYWORD_INDEX:
        if keyword in q:
            return True
    return len(q.split()) >= 5


# ── Web-fallback thresholds ───────────────────────────────────────────────────
WEB_FALLBACK_MIN_HITS  = 2
WEB_FALLBACK_MIN_SCORE = 0.55
WEB_FALLBACK_AVG_SCORE = 0.50
WEB_FALLBACK_MAX_PAGES = 4
WEB_FALLBACK_DELAY     = 0.8


# ─────────────────────────────────────────────
# MULTI-SOURCE WEB SEARCH
# ─────────────────────────────────────────────

_SEARCH_TIMEOUT   = 5       # reduced from 10 — stalled backends cut faster
_PARALLEL_WORKERS = 3

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_DOMAIN_TRUST: dict[str, float] = {
    "github.com": 2.0,
    "gitlab.com": 1.8,
    "raw.githubusercontent.com": 1.5,
    "docs.python.org": 2.0,
    "man7.org": 1.8,
    "linux.die.net": 1.8,
    "kernel.org": 2.0,
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
    "stackoverflow.com": 1.5,
    "superuser.com": 1.2,
    "askubuntu.com": 1.2,
    "debian.org": 1.8,
    "archlinux.org": 1.8,
    "redhat.com": 1.5,
    "ubuntu.com": 1.5,
    "krebsonsecurity.com": 1.5,
    "schneier.com": 1.5,
    "theregister.com": 1.2,
    "bleepingcomputer.com": 1.5,
    "securityweek.com": 1.3,
    "sans.org": 1.8,
    "cisco.com": 1.3,
    "paloaltonetworks.com": 1.3,
    "tryhackme.com": 1.5,
    "hackthebox.com": 1.5,
    "ctftime.org": 1.5,
}

_DOMAIN_PENALTY: dict[str, float] = {
    "pinterest.com": -5.0,
    "pinterest.co.uk": -5.0,
    "quora.com": -2.0,
    "medium.com": -0.5,
    "reddit.com": -0.5,
    "scribd.com": -3.0,
    "slideshare.net": -2.0,
    "chegg.com": -3.0,
    "coursehero.com": -3.0,
    "answers.yahoo.com": -4.0,
}

_DOMAIN_BLOCKLIST: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be",
    "amazon.com", "ebay.com", "etsy.com",
    "yelp.com", "tripadvisor.com",
    "duckduckgo.com", "bing.com", "google.com", "mojeek.com",
}

# Path signals — compiled into single alternations for a fast pre-check,
# then individual patterns only run when the combined one matches.
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
_PATH_BONUS_RE   = re.compile("|".join(p.pattern for p, _ in _PATH_BONUS_RULES),   re.I)
_PATH_PENALTY_RE = re.compile("|".join(p.pattern for p, _ in _PATH_PENALTY_RULES), re.I)


def _strip_www(domain: str) -> str:
    return domain[4:] if domain.startswith("www.") else domain


def _fetch_html(url: str, timeout: int = _SEARCH_TIMEOUT) -> str:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("_fetch_html failed for %s: %s", url, exc)
        return ""


def _extract_hrefs(html: str) -> list[str]:
    urls = []
    for m in re.finditer(r'href=["\']?(https?://[^"\'>\s]+)', html):
        u = _html.unescape(m.group(1))
        if u not in urls:
            urls.append(u)
    return urls


def _search_ddg(query: str, max_results: int) -> list[str]:
    encoded = urllib.parse.quote_plus(query)
    body = _fetch_html(f"https://html.duckduckgo.com/html/?q={encoded}")
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
    encoded = urllib.parse.quote_plus(query)
    body = _fetch_html(f"https://www.bing.com/search?q={encoded}&count={max_results * 2}")
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
    encoded = urllib.parse.quote_plus(query)
    body = _fetch_html(f"https://www.mojeek.com/search?q={encoded}&l={max_results * 2}")
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


def _score_url(url: str) -> float:
    # Cheap string pre-check before paying for urlparse
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

    if any(domain == b or domain.endswith("." + b) for b in _DOMAIN_BLOCKLIST):
        return -99.0

    score = 0.0

    for trusted, bonus in _DOMAIN_TRUST.items():
        if domain == trusted or domain.endswith("." + trusted):
            score += bonus
            break

    for penalised, penalty in _DOMAIN_PENALTY.items():
        if domain == penalised or domain.endswith("." + penalised):
            score += penalty
            break

    if _PATH_BONUS_RE.search(path):
        for pattern, bonus in _PATH_BONUS_RULES:
            if pattern.search(path):
                score += bonus

    if _PATH_PENALTY_RE.search(path):
        for pattern, penalty in _PATH_PENALTY_RULES:
            if pattern.search(path):
                score += penalty

    if parsed.scheme != "https":
        score -= 0.5

    depth = path.count("/")
    if depth <= 3:
        score += 0.3
    elif depth >= 7:
        score -= 0.3

    return score


def _multi_search(query: str, max_results: int = 6) -> list[str]:
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

    seen_keys: set[str] = set()
    scored: list[tuple[float, str]] = []
    for url in raw_urls:
        score = _score_url(url)
        if score < -10:
            continue
        try:
            parsed = urllib.parse.urlparse(url)
            key = _strip_www(parsed.netloc.lower()) + parsed.path.rstrip("/").lower()
        except Exception:
            key = url
        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored.append((score, url))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = [url for _, url in scored[:max_results]]
    logger.info("_multi_search: %d raw → %d unique → top %d for %r",
                len(raw_urls), len(scored), len(top), query)
    return top


# ─────────────────────────────────────────────
# FETCH + CLEAN
# ─────────────────────────────────────────────

# Import optional heavy dependencies once at module level so repeated calls to
# _fetch_and_clean don't pay the import overhead every time.
try:
    import trafilatura as _trafilatura
except ImportError:
    _trafilatura = None

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
except ImportError:
    _BeautifulSoup = None

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


def _fetch_and_clean(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers=_FETCH_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            if "text" not in resp.headers.get("Content-Type", ""):
                return ""
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Fallback fetch failed for %s: %s", url, exc)
        return ""

    text = ""

    if _trafilatura is not None:
        text = _trafilatura.extract(raw_html) or ""

    if (not text or len(text) < 200) and _BeautifulSoup is not None:
        try:
            soup = _BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer"]):
                tag.decompose()
            main = soup.find("article") or soup.find("main") or soup.body
            text = "\n".join(main.stripped_strings) if main else ""
        except Exception:
            pass

    clean_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) < 4:
            continue
        letters = sum(1 for c in s if c.isalpha())
        if len(s) > 0 and letters / len(s) < 0.4:
            continue
        clean_lines.append(s)

    return "\n".join(clean_lines)


def _build_signal_from_url(url: str, content: str = "") -> str:
    raw = unquote(url)
    if "github.com" in raw:
        m = re.search(r"/(?:blob|tree)/[^/]+/(.+)$", raw)
        path_part = m.group(1) if m else raw.split("/")[-1]
    else:
        p_parsed = urlparse(raw)
        path_part = p_parsed.path.rstrip("/")
        if "/" in path_part:
            parts = [p for p in path_part.split("/") if p]
            path_part = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else url)

    signal = re.sub(r"[/_\-%.]+", " ", path_part).strip()
    for line in content.splitlines():
        s = line.strip().lstrip("#").strip()
        if len(s) >= 8 and re.search(r"[a-zA-Z]{3,}", s):
            signal = signal + " " + s[:120]
            break
    return signal.lower()


def extract_core_entity(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"^(what is|who is|tell me about)\s+", "", q)
    q = re.sub(r"\b(anime|manga|series|show|tv|film|movie)\b", "", q)
    q = re.sub(r"[^a-z0-9\s]", "", q)
    return " ".join(q.split()).strip()


def enhance_query(query: str) -> str:
    return f"{extract_core_entity(query)} wikipedia official summary"


def is_valid_source(query: str, url: str, text: str) -> bool:
    core_tokens = extract_core_entity(query).split()
    url_l  = url.lower()
    text_l = text.lower()
    if core_tokens and not any(tok in url_l or tok in text_l for tok in core_tokens):
        return False
    words = text.split()
    if len(words) < 200:
        return False
    if len(set(words)) < 80:
        return False
    if words.count("http") > 25:
        return False
    return True


# ─────────────────────────────────────────────
# VECTOR SEARCH HELPERS
# ─────────────────────────────────────────────

def _raw_vector_search(
    qvec: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    conn = get_chroma_collection()
    try:
        cur = conn.cursor()
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
    finally:
        _return_conn(conn)

    return [
        {
            "id": row_id, "text": text, "source": source,
            "score": round(1 / (1 + dist), 4),
            "category": category, "subcategory": subcategory,
            "chunk_index": chunk_index,
        }
        for row_id, text, source, category, subcategory, chunk_index, dist in rows
        if 1 / (1 + dist) >= min_score
    ]


def _raw_vector_search_category(
    qvec: list[float],
    category: str,
    top_k: int,
    min_score: float,
) -> list[dict]:
    conn = get_chroma_collection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index,
                   embedding <-> %s::vector AS distance
            FROM documents
            WHERE category = %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (qvec, category, qvec, top_k),
        )
        rows = cur.fetchall()
    except Exception:
        conn.rollback()
        raise
    finally:
        _return_conn(conn)

    return [
        {
            "id": row_id, "text": text, "source": source,
            "score": round(1 / (1 + dist), 4),
            "category": category, "subcategory": subcategory,
            "chunk_index": chunk_index,
        }
        for row_id, text, source, category, subcategory, chunk_index, dist in rows
        if 1 / (1 + dist) >= min_score
    ]


# ─────────────────────────────────────────────
# WEB FALLBACK INGEST
# ─────────────────────────────────────────────
def extract_core_entity(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"^(what is|who is|tell me about)\s+", "", q)
    q = re.sub(r"\b(anime|manga|series|show|tv|film|movie)\b", "", q)
    q = re.sub(r"[^a-z0-9\s]", "", q)
    return " ".join(q.split()).strip()


def _web_fallback_ingest(query: str) -> int:
    logger.info("RAG web fallback triggered for query: %r", query)
    enhanced = enhance_query(query)
    core = extract_core_entity(query)

    candidate_urls = _multi_search(enhanced, max_results=WEB_FALLBACK_MAX_PAGES + 5)
    if not candidate_urls:
        logger.warning("Web fallback: no results returned")
        return 0

    total_chunks = 0
    pages_used   = 0
    best_url     = None
    best_count   = 0

    # Split core into tokens; for short titles use ANY match, longer use ALL
    raw_tokens  = core.split()
    token_match = any if len(raw_tokens) <= 2 else all

    for url in candidate_urls:
        if pages_used >= WEB_FALLBACK_MAX_PAGES:
            break

        time.sleep(WEB_FALLBACK_DELAY)
        text = _fetch_and_clean(url)
        if not text:
            continue

        if not is_valid_source(query, url, text):
            logger.debug("Rejected source (failed grounding): %s", url)
            continue

        # Page-level relevance: count total token hits across the page
        text_l    = text.lower()
        relevance = sum(text_l.count(tok) for tok in raw_tokens)
        if relevance < 10:
            logger.debug("Rejected source (low relevance score %d): %s", relevance, url)
            continue

        chunks      = chunk_text(text)
        seen        = set()
        good_chunks = []
        for c in chunks:
            norm = " ".join(c.lower().split())
            if norm in seen or len(c.split()) < 10:
                continue
            if not token_match(tok in c.lower() for tok in raw_tokens):
                continue
            seen.add(norm)
            good_chunks.append(c)

        if not good_chunks:
            continue

        n = upsert_chunks(good_chunks, source=url)
        total_chunks += n
        pages_used   += 1

        if n > best_count:
            best_count = n
            best_url   = url

        logger.info("Web fallback: +%d chunks from %s (relevance=%d)", n, url, relevance)

    if best_url:
        logger.info("Best source: %s (%d chunks)", best_url, best_count)

    logger.info("Web fallback complete: %d chunks from %d pages", total_chunks, pages_used)
    return total_chunks


# ─────────────────────────────────────────────
# MAIN RETRIEVAL
# ─────────────────────────────────────────────

# Rich console cached at module level — avoids re-importing on every call
try:
    from rich.console import Console as _Console
    _console_print = _Console().print
except ImportError:
    _console_print = print


def retrieve_context(
    query: str,
    top_k: int = config.TOP_K_RESULTS,
    min_score: float = config.MIN_RELEVANCE_SCORE,
    *,
    web_fallback: bool = True,
):
    """
    Retrieve relevant chunks for *query*.

    Search order:
        1. Normal vector search (all categories)
        2. If weak, try category='other'
        3. If still weak, web fallback
        4. Re-run vector search after ingest
    """
    q = query.strip()
    if not q or q.lower() in RAG_SKIP_PHRASES or len(q) < RAG_MIN_LENGTH:
        return []

    embedder = get_embedder()
    qvec     = embedder.encode(query).tolist()

    # ── Pass 1: normal search ─────────────────────────────────────────────────
    hits = _raw_vector_search(qvec, top_k, min_score)

    def _is_weak(hits_: list[dict]) -> bool:
        if len(hits_) < WEB_FALLBACK_MIN_HITS:
            return True
        if hits_ and hits_[0]["score"] < WEB_FALLBACK_MIN_SCORE:
            return True
        avg = sum(h["score"] for h in hits_) / len(hits_) if hits_ else 0.0
        return avg < WEB_FALLBACK_AVG_SCORE

    # ── Pass 2: 'other' category fallback ────────────────────────────────────
    used_other = False
    if _is_weak(hits):
        # Reuse _raw_vector_search_category instead of duplicating SQL inline
        other_hits = _raw_vector_search_category(qvec, "other", top_k, min_score)
        if other_hits:
            logger.info("Using %d hit(s) from category='other'", len(other_hits))
            hits       = other_hits
            used_other = True

    # ── Pass 3: web fallback ──────────────────────────────────────────────────
    used_web = False
    if web_fallback and (_is_weak(hits) or (not is_retrieval_query(query) and len(q.split()) >= 4)):
        logger.info("Local retrieval weak; triggering web fallback")
        new_chunks = _web_fallback_ingest(query)
        if new_chunks > 0:
            hits     = _raw_vector_search(qvec, top_k, min_score)
            used_web = True
        else:
            _console_print("[dim yellow]⟳ RAG web fallback triggered but found nothing useful[/]")

    # ── Annotate provenance ───────────────────────────────────────────────────
    for h in hits:
        h["web_fallback"] = used_web

    return hits


# ─────────────────────────────────────────────
# CONTEXT FORMATTING
# ─────────────────────────────────────────────

def format_context_block(rag_hits: list[dict]) -> str:
    if not rag_hits:
        return ""

    pg_hits  = [h for h in rag_hits if not h.get("web_fallback")]
    web_hits = [h for h in rag_hits if h.get("web_fallback")]

    if pg_hits and web_hits:
        preamble = (
            "The following context was retrieved to help answer the query. "
            f"{len(pg_hits)} chunk(s) came from PostgreSQL "
            f"and {len(web_hits)} chunk(s) were fetched live from the web because PostgreSQL "
            "had insufficient coverage for this query. "
            "Web-sourced chunks are unvetted — treat them as helpful but potentially "
            "incomplete or inaccurate. Prefer PostgreSQL chunks where they conflict."
        )
    elif web_hits:
        preamble = (
            "The following context was retrieved to help answer the query. "
            f"All {len(web_hits)} chunk(s) were fetched live from the web — "
            "PostgreSQL had no relevant content for this query. "
            "These chunks are unvetted. Cross-check claims where accuracy is critical."
        )
    else:
        preamble = (
            "The following context was retrieved to help answer the query. "
            f"All {len(pg_hits)} chunk(s) came from PostgreSQL."
        )

    blocks = [
        f"[SOURCE: {h.get('source', 'unknown')} | "
        f"cat: {h.get('category', 'unknown')}/{h.get('subcategory', 'unknown')} | "
        f"chunk: {h.get('chunk_index', '?')} | "
        f"id: {h.get('id', '?')} | "
        f"score: {h.get('score', '?')} | "
        f"via: {'web' if h.get('web_fallback') else 'postgresql'}]\n"
        f"{h.get('text', '')}"
        for h in rag_hits
    ]

    return preamble + "\n\n" + "\n\n".join(blocks)


# ─────────────────────────────────────────────
# COUNTS / LISTING
# ─────────────────────────────────────────────

def get_db_count() -> int:
    conn = get_chroma_collection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]
    finally:
        _return_conn(conn)


def count_documents() -> int:
    return get_db_count()


def list_categories(collection=None) -> list[str]:
    conn = get_chroma_collection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT category FROM documents")
        return [r[0] for r in cur.fetchall()]
    finally:
        _return_conn(conn)


def get_by_category(category: str, subcategory: str = None, collection=None) -> list[dict]:
    conn = get_chroma_collection()
    try:
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
    finally:
        _return_conn(conn)