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
import requests

import config
from config import configure_logging
from rag import get_chroma_collection, retrieve_context, format_context_block, get_db_count

########################
#hacker
from hack import connection
########################

configure_logging()
logger = logging.getLogger(__name__)
console = Console()

Message = dict[str, str]

# RAG modes
RAG_MODE_TOOL   = "tool"    # model decides when to search (tool calling)
RAG_MODE_INJECT = "inject"  # always retrieve + inject before sending
RAG_MODE_OFF    = "off"     # no RAG at all


# ──────────────────────────────────────────────
# MEMORY
# ──────────────────────────────────────────────

import importlib.util
import inspect


class LazyAddons:
    def __init__(self, folder="addons"):
        self.paths = {}
        self.cache = {}

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".py"):
                    name = os.path.splitext(file)[0]
                    self.paths[name] = os.path.join(root, file)

    def __contains__(self, name):
        return name in self.paths

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]

        path = self.paths[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func = getattr(module, name)
        self.cache[name] = func
        return func


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
# TOOL DEFINITIONS
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the local knowledge base for relevant information. "
                "Only call this when the user asks about a specific topic "
                "that may be in your documents. Do NOT call for greetings, "
                "casual chat, or questions you already know the answer to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the knowledge base",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def run_search_tool(query: str, web_search: bool = False) -> str:
    """Execute the search_docs tool and return formatted results as a string."""
    if not isinstance(query, str):
        query = str(query)
    query = query.strip()
    if not query:
        return "No query provided."
    try:
        hits = retrieve_context(
            query,
            top_k=config.TOP_K_RESULTS,
            min_score=config.MIN_RELEVANCE_SCORE,
            web_fallback=web_search,
        )
        if not hits:
            return "No relevant information found in the knowledge base."

        result = format_context_block(hits)
        if not isinstance(result, str):
            result = str(result)

        db_hits  = [h for h in hits if not h.get("web_fallback")]
        web_hits = [h for h in hits if h.get("web_fallback")]

        if db_hits:
            sources = {h["source"].split("/")[-1] for h in db_hits}
            console.print(f"[red]📎 DATABASE CHUNK: {len(db_hits)} chunk(s) from: {', '.join(sources)}[/]")
        if web_hits:
            sources = {h["source"].split("/")[-1] for h in web_hits}
            console.print(f"[red]📎 WEB CHUNK: {len(web_hits)} chunk(s) from: {', '.join(sources)}[/]")

        return result
    except Exception as exc:
        logger.error("Tool search failed: %s", exc)
        return f"Search failed: {exc}"


# ──────────────────────────────────────────────
# STREAMING
# ──────────────────────────────────────────────

async def stream_response_tool(
    client: AsyncClient,
    messages: list[Message],
    model: str,
    web_search: bool = False,
) -> AsyncIterator[str]:
    """Tool-calling mode: model decides when to search."""
    response = await client.chat(
        model=model,
        messages=messages,
        stream=False,
        options=config.OLLAMA_OPTIONS,
        tools=TOOLS,
    )

    msg = response.message

    if msg.tool_calls:
        tool_messages = list(messages)
        tool_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls,
        })

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments or {}

            if fn_name == "search_docs":
                query = fn_args.get("query", "") if isinstance(fn_args, dict) else str(fn_args)
                tool_result = await asyncio.to_thread(run_search_tool, query, web_search)
            else:
                tool_result = f"Unknown tool: {fn_name}"

            tool_messages.append({"role": "tool", "content": tool_result})

        async for chunk in await client.chat(
            model=model,
            messages=tool_messages,
            stream=True,
            options=config.OLLAMA_OPTIONS,
        ):
            token = chunk.message.content
            if token:
                yield token
    else:
        async for chunk in await client.chat(
            model=model,
            messages=messages,
            stream=True,
            options=config.OLLAMA_OPTIONS,
        ):
            token = chunk.message.content
            if token:
                yield token


async def stream_response_inject(
    client: AsyncClient,
    messages: list[Message],
    model: str,
    raw_query: str,
    web_search: bool = False,
) -> AsyncIterator[str]:
    """Inject mode: always retrieve and prepend context before sending."""
    rag_hits: list[dict] = []
    try:
        rag_hits = await asyncio.to_thread(
            retrieve_context,
            raw_query,
            top_k=config.TOP_K_RESULTS,
            min_score=config.MIN_RELEVANCE_SCORE,
            web_fallback=web_search,
        )
        if rag_hits:
            db_hits  = [h for h in rag_hits if not h.get("web_fallback")]
            web_hits = [h for h in rag_hits if h.get("web_fallback")]

            if db_hits:
                sources = {h["source"].split("/")[-1] for h in db_hits}
                console.print(f"[bold red]📎 DATABASE CHUNK:[/] [red]{len(db_hits)} chunk(s) from: {', '.join(sources)}[/]")
            if web_hits:
                sources = {h["source"].split("/")[-1] for h in web_hits}
                console.print(f"[bold red]📎 WEB CHUNK:[/] [red]{len(web_hits)} chunk(s) from: {', '.join(sources)}[/]")

    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)

    # Swap the last user message with the RAG-augmented version
    context_block = format_context_block(rag_hits)
    augmented_messages = list(messages)
    if context_block and augmented_messages:
        last = augmented_messages[-1]
        if last["role"] == "user":
            augmented_messages[-1] = {
                "role": "user",
                "content": f"{context_block}\n\nUser question: {last['content']}",
            }

    async for chunk in await client.chat(
        model=model,
        messages=augmented_messages,
        stream=True,
        options=config.OLLAMA_OPTIONS,
    ):
        token = chunk.message.content
        if token:
            yield token


async def stream_response_off(
    client: AsyncClient,
    messages: list[Message],
    model: str,
) -> AsyncIterator[str]:
    """No RAG — plain streaming."""
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
# TRIM MESSAGES
# ──────────────────────────────────────────────

def trim_messages(messages: list[dict], max_tokens: int = 4096) -> list[dict]:
    """Keep system prompt + trim oldest exchanges to stay under token budget."""
    system = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    def tokens(m):
        return len(str(m.get("content", ""))) // 4

    system_tokens = sum(tokens(m) for m in system)
    budget = max_tokens - system_tokens

    while conversation and sum(tokens(m) for m in conversation) > budget:
        conversation = conversation[2:]

    return system + conversation


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
    "/rag":    "Switch RAG mode  (/rag tool | /rag inject | /rag off)",
    "/web":    "Toggle web search  (/web on | /web off)",
    "/exit":   "Quit the assistant",
}


def handle_command(
    command: str,
    memory: ConversationMemory,
    state: dict,
) -> bool:
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/help":
        table = Table(title="Available Commands", show_header=True, header_style="bold red")
        table.add_column("Command", style="red")
        table.add_column("Description")
        for c, desc in COMMANDS.items():
            table.add_row(c, desc)
        console.print(table)

    elif cmd == "/rag":
        valid = (RAG_MODE_TOOL, RAG_MODE_INJECT, RAG_MODE_OFF)
        if arg not in valid:
            console.print(
                f"[yellow]Current RAG mode: [bold]{state['rag_mode']}[/]\n"
                f"Usage: /rag tool | /rag inject | /rag off[/]"
            )
            return True
        state["rag_mode"] = arg
        descriptions = {
            RAG_MODE_TOOL:   "model decides when to search (tool calling)",
            RAG_MODE_INJECT: "always retrieve + inject context before every message",
            RAG_MODE_OFF:    "no RAG — model uses only its own knowledge",
        }
        console.print(f"[green]✓ RAG mode → [bold]{arg}[/] — {descriptions[arg]}[/]")

    elif cmd == "/web":
        if arg in ("on", "off"):
            state["web_search"] = arg == "on"
            status = "[green]enabled[/]" if state["web_search"] else "[red]disabled[/]"
            console.print(f"[green]✓ Web search → {status}[/]")
        else:
            current = "on" if state["web_search"] else "off"
            console.print(
                f"[yellow]Web search is currently: [bold]{current}[/]\n"
                f"Usage: /web on | /web off[/]"
            )

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
        table.add_row("RAG mode", state["rag_mode"])
        table.add_row("Web search", "on" if state["web_search"] else "off")
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
        table.add_column("Value", style="red")
        table.add_row("Model", state["model"])
        table.add_row("RAG mode", state["rag_mode"])
        table.add_row("Web search", "on" if state["web_search"] else "off")
        table.add_row("Ollama host", config.OLLAMA_HOST)
        table.add_row("History pairs", f"{memory.pair_count} / {memory.max_pairs}")
        table.add_row("Top-K retrieval", str(config.TOP_K_RESULTS))
        table.add_row("Embedding model", config.EMBEDDING_MODEL)
        console.print(table)

    elif cmd in ("/exit", "/quit", "/bye"):
        console.print("\n[bold green]Goodbye! 👋[/]")
        return False

    elif cmd.startswith("/"):
        finalcmd = cmd.replace("/", "")
        argsplit = parts[1].split() if len(parts) > 1 else []
        addons = LazyAddons("addons")
        print(f"Looking for: {finalcmd}")
        if finalcmd in addons:
            try:
                result = addons[finalcmd](*argsplit)
                if isinstance(result, str):
                    console.rule("[bold red]Addon output[/]")
                    console.print(f"[white]{result}[/]")
                    console.rule("[bold red][/]")
                    console.print("[bold red]Would you like to send to ai?[bold yellow]Y[/]/[bold yellow]N[/][/]")
                    userconfirm = input("\u200B")
                    if userconfirm == "y":
                        return result
                    else:
                        return
            except TypeError as e:
                print(f"Argument error for '{finalcmd}': {e}    args:{argsplit}")
        else:
            console.print(f"[red]Unknown command '{cmd}'. Type /help for options.[/]")

    return True


# ──────────────────────────────────────────────
# MAIN CHAT LOOP
# ──────────────────────────────────────────────

async def chat_loop() -> None:
    client = AsyncClient(host=config.OLLAMA_HOST)
    memory = ConversationMemory()
    state = {
        "model": config.DEFAULT_MODEL,
        "rag_mode": RAG_MODE_INJECT,  # default mode
        "web_search": False,          # web search disabled by default
    }

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
        f"[bold red]  RAG     [/][white]{state['rag_mode']}[/]\n"
        f"[bold red]  Web     [/][white]{'on' if state['web_search'] else 'off'}[/]\n"
        "[dim]──────────────────────────────────────────────────────────────────────────────[/]"
        + db_section +
        "[red]──────────────────────────────────────────────────────────────────────────────[/]\n"
        "  [white]/help[/] [red]·[/] [white]/model[/] [red]·[/] [white]/list[/] [red]·[/] [white]/docs[/] [red]·[/] [white]/info[/] [red]·[/] [white]/clear[/] [red]·[/] [white]/rag[/] [red]·[/] [white]/web[/] [red]·[/] [white]/exit[/]",
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
            result = handle_command(raw, memory, state)
            if result is False:
                break
            if isinstance(result, str):
                raw = result
            else:
                continue

        # ── Build base messages ──
        user_msg: Message = {"role": "user", "content": raw}
        messages_to_send = trim_messages(memory.get_messages(config.SYSTEM_PROMPT) + [user_msg])

        console.print(Rule(style="bold red"))

        full_response = ""
        mode = state["rag_mode"]
        web_search = state["web_search"]

        try:
            with console.status("[red]thinking…[/]", spinner="dots", spinner_style="bold red"):
                if mode == RAG_MODE_TOOL:
                    gen = stream_response_tool(client, messages_to_send, state["model"], web_search)
                elif mode == RAG_MODE_INJECT:
                    gen = stream_response_inject(client, messages_to_send, state["model"], raw, web_search)
                else:  # off
                    gen = stream_response_off(client, messages_to_send, state["model"])

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
        logger.info("Exchange saved. Pairs: %d. Mode: %s.", memory.pair_count, mode)


def main() -> None:
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye! 👋[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
