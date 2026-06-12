"""
main.py — Streaming CLI chat interface for the local AI assistant.
Everything is lazy: DB connection and embedder only load on first query.
"""

from __future__ import annotations

from datetime import datetime
import asyncio
import logging
import sys
import os
import warnings
from collections import deque
from typing import AsyncIterator

# ── Suppress PyTorch / HuggingFace noise before any imports ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import ollama
from ollama import AsyncClient, ResponseError
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import config
from config import configure_logging
from rag import get_chroma_collection, retrieve_context, format_context_block, get_db_count

configure_logging()
logger = logging.getLogger(__name__)
console = Console()

Message = dict[str, str]


# ──────────────────────────────────────────────
# MEMORY
# ──────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_pairs: int = config.MAX_HISTORY_PAIRS) -> None:
        self.max_pairs = max_pairs
        self._pairs: deque[tuple[Message, Message]] = deque(maxlen=max_pairs)

    def add_exchange(self, user_msg: Message, assistant_msg: Message) -> None:
        self._pairs.append((user_msg, assistant_msg))

    def get_messages(self, system_prompt: str) -> list[Message]:
        msgs: list[Message] = [{"role": "system", "content": system_prompt}]
        for user, assistant in self._pairs:
            msgs.append(user)
            msgs.append(assistant)
        return msgs

    def clear(self) -> None:
        self._pairs.clear()

    @property
    def pair_count(self) -> int:
        return len(self._pairs)


# ──────────────────────────────────────────────
# STREAMING
# ──────────────────────────────────────────────

async def stream_response(
    client: AsyncClient,
    messages: list[Message],
    model: str,
) -> AsyncIterator[str]:
    async for chunk in await client.chat(
        model=model,
        messages=messages,
        stream=True,
        options=config.OLLAMA_OPTIONS,
    ):
        token = chunk.message.content
        if token:
            yield token


# ──────────────────────────────────────────────
# RAG MESSAGE BUILDER
# ──────────────────────────────────────────────

def build_user_message(query: str, rag_hits: list[dict]) -> Message:
    context_block = format_context_block(rag_hits)
    if context_block:
        content = f"{context_block}\n\nUser question: {query}"
    else:
        content = query
    return {"role": "user", "content": content}


# ──────────────────────────────────────────────
# SLASH COMMANDS
# ──────────────────────────────────────────────

COMMANDS = {
    "/help":   "Show this help message",
    "/model":  "Switch the active model  (/model mistral  or  /model 2)",
    "/list":   "List available models",
    "/clear":  "Clear conversation history",
    "/docs":   "Show how many vectors are in the store",
    "/status": "Show current config / model",
    "/info":   "System information",
    "/exit":   "Quit the assistant",
}


def handle_command(
    command: str,
    memory: ConversationMemory,
    state: dict,
) -> bool:
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        table = Table(title="Available Commands", show_header=True, header_style="bold red")
        table.add_column("Command", style="red")
        table.add_column("Description")
        for c, desc in COMMANDS.items():
            table.add_row(c, desc)
        console.print(table)

    elif cmd == "/info":
        response = ollama.list()
        models = [m.get("model") for m in response.get("models", [])]
        try:
            doc_count = get_db_count()
        except Exception:
            doc_count = "unavailable"

        table = Table(title="System Information", show_header=False)
        table.add_column("Key", style="bold red")
        table.add_column("Value", style="red")
        table.add_row("Active model", state["model"])
        table.add_row("Available models", ", ".join(models) if models else "None")
        table.add_row("Ollama host", config.OLLAMA_HOST)
        table.add_row("Conversation pairs", f"{memory.pair_count}/{memory.max_pairs}")
        table.add_row("Vector DB chunks", str(doc_count))
        table.add_row("RAG top_k", str(config.TOP_K_RESULTS))
        table.add_row("Embedding model", config.EMBEDDING_MODEL)
        table.add_row("Streaming", "enabled")
        console.print(table)

    elif cmd == "/clear":
        memory.clear()
        console.print("[yellow]🗑  Conversation history cleared.[/]")

    elif cmd == "/model":
        models = [m.get("model") for m in ollama.list().get("models", [])]
        if not arg:
            console.print(f"[yellow]Current model: [bold]{state['model']}[/][/]")
            return True
        arg = arg.strip()
        if arg.isdigit():
            idx = int(arg) - 1
            if idx < 0 or idx >= len(models):
                console.print("[red]Invalid model number[/]")
                return True
            state["model"] = models[idx]
        else:
            state["model"] = arg
        console.print(f"[green]✓ Model switched to [bold]{state['model']}[/][/]")

    elif cmd == "/list":
        response = ollama.list()
        console.rule("[bold red]MODELS[/]")
        for i, model in enumerate(response.get("models", []), start=1):
            console.print(f"{i}. {model.get('model')}")
        console.rule()

    elif cmd == "/docs":
        try:
            count = get_db_count()
            console.print(f"[red]📚 Vector store contains [bold]{count}[/] chunks.[/]")
        except Exception as exc:
            console.print(f"[red]Could not query vector store: {exc}[/]")

    elif cmd == "/status":
        table = Table(title="Assistant Status", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value", style="cyan")
        table.add_row("Model", state["model"])
        table.add_row("Ollama host", config.OLLAMA_HOST)
        table.add_row("History pairs", f"{memory.pair_count} / {memory.max_pairs}")
        table.add_row("Top-K retrieval", str(config.TOP_K_RESULTS))
        table.add_row("Embedding model", config.EMBEDDING_MODEL)
        console.print(table)

    elif cmd in ("/exit", "/quit", "/bye"):
        console.print("\n[bold green]Goodbye! 👋[/]")
        return False

    else:
        console.print(f"[red]Unknown command '{cmd}'. Type /help for options.[/]")

    return True


# ──────────────────────────────────────────────
# MAIN CHAT LOOP
# ──────────────────────────────────────────────

async def chat_loop() -> None:
    client = AsyncClient(host=config.OLLAMA_HOST)
    memory = ConversationMemory()
    state = {"model": config.DEFAULT_MODEL}

    # ── fetch DB stats for banner ──
    try:
        from rag import get_db_count, list_categories, get_by_category
        doc_count = get_db_count()
        cats = list_categories()
        cat_lines = ""
        for cat in cats:
            items = get_by_category(cat)
            subcats = sorted({it.get("subcategory", "none") for it in items})
            words = sum(len(it["text"].split()) for it in items)
            cat_lines += (
                f"\n  [red]·[/] [bold]{cat}[/]  "
                f"[red]chunks = [/][white]{len(items)}[/]  "
                f"[red]words = [/][white]{words}[/]  "
                f"[red]sub = [/][white]{', '.join(subcats)}[/]"
            )
        db_section = (
            f"\n[bold red]  Chunks  [/][white]{doc_count}[/]"
            f"\n[bold red]  Categories[/]{cat_lines}\n"
        )
    except Exception:
        db_section = "\n[dim]  DB unavailable[/]\n"

    console.print(Panel(
        "[dim]──────────────────────────────────────────────────────────────────────────────[/]\n"
        f"[bold red]  Model   [/][white]{state['model']}[/]\n"
        f"[bold red]  Host    [/][white]{config.OLLAMA_HOST}[/]\n"
        "[dim]──────────────────────────────────────────────────────────────────────────────[/]"
        + db_section +
        "[red]──────────────────────────────────────────────────────────────────────────────[/]\n"
        "  [white]/help[/] [red]·[/] [white]/model[/] [red]·[/] [white]/list[/] [red]·[/] [white]/docs[/] [red]·[/] [white]/info[/] [red]·[/] [white]/clear[/] [red]·[/] [white]/exit[/]",
        border_style="bold red",
        padding=(1, 4),
        width=console.width,
    ))
    console.print()

    while True:
        try:
            raw = Prompt.ask(f"[bold red][{datetime.now().strftime('%H:%M')}][/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye! 👋[/]")
            break

        raw = raw.strip()
        if not raw:
            continue

        if raw.startswith("/"):
            should_continue = handle_command(raw, memory, state)
            if not should_continue:
                break
            continue

        # ── RAG retrieval ──
        rag_hits: list[dict] = []
        try:
            rag_hits = retrieve_context(
                raw,
                top_k=config.TOP_K_RESULTS,
                min_score=config.MIN_RELEVANCE_SCORE,
            )
            if rag_hits:
                sources = {h["source"].split("/")[-1] for h in rag_hits}
                console.print(
                    f"[dim]📎 Retrieved {len(rag_hits)} chunk(s) from: "
                    f"{', '.join(sources)}[/]"
                )
        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc)

        # ── Build messages ──
        user_msg = build_user_message(raw, rag_hits)
        messages_to_send = memory.get_messages(config.SYSTEM_PROMPT) + [user_msg]

        console.print(Rule(style="bold red"))

        # ── Thinking spinner until first token arrives ──
        full_response = ""
        first_token = True

        async def get_first_token():
            async for token in stream_response(client, messages_to_send, state["model"]):
                return token
            return ""

        try:
            with console.status("[red]thinking…[/]", spinner="dots", spinner_style="bold red"):
                # collect first token so spinner shows until model starts responding
                gen = stream_response(client, messages_to_send, state["model"])
                first = await gen.__anext__()

            console.print(Text("AI:", style="bold red"))

            with Live(
                Markdown(first + "▋"),
                console=console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                full_response = first
                async for token in gen:
                    full_response += token
                    live.update(Markdown(full_response + "▋"))
                live.update(Markdown(full_response))

        except StopAsyncIteration:
            pass
        except ResponseError as exc:
            console.print(f"\n[red]Ollama error: {exc}[/]")
            if "model" in str(exc).lower():
                console.print(
                    f"[yellow]Tip: run [bold]ollama pull {state['model']}[/] to download the model.[/]"
                )
            logger.error("Ollama ResponseError: %s", exc)
            continue
        except Exception as exc:
            console.print(f"\n[red]Unexpected error: {exc}[/]")
            logger.exception("Unexpected error during streaming.")
            continue

        console.print(Rule(style="bold red"))
        console.print()

        memory.add_exchange(
            {"role": "user", "content": raw},
            {"role": "assistant", "content": full_response},
        )
        logger.info(
            "Exchange saved. Pairs: %d. RAG hits: %d.",
            memory.pair_count,
            len(rag_hits),
        )


def main() -> None:
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye! 👋[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()