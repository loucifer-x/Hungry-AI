# Hungry AI

**Feed it data. It remembers.**

Hungry AI is a local AI agent powered by Ollama LLMs that becomes more personalized with every piece of information you provide.

Give it your documents, notes, PDFs, websites, codebases, research, and knowledge sources. Hungry AI scans, crawls, indexes, and remembers what it learns, building a private knowledge base tailored specifically to you.

Your AI becomes more useful over time—understanding your projects, interests, workflows, and domain knowledge instead of relying solely on a generic language model.

> **Your data stays your data.** Everything runs locally and remains under your control.

---

## Features

* 🖥️ Runs locally using Ollama LLMs
* 📄 Ingests documents, notes, and PDFs
* 🌐 Crawls websites and online resources
* 🧠 Builds a personalized knowledge base from your data
* 🔍 Retrieves relevant information when answering questions
* 🔗 Combines retrieval with LLM reasoning (RAG)
* 📚 Learns from both local files and web content
* 🔒 Keeps your data on your machine

---

## How It Works

Hungry AI uses **Retrieval-Augmented Generation (RAG)** to search, retrieve, and reason over external knowledge sources before generating responses.

Instead of depending only on an LLM's built-in knowledge, Hungry AI can:

1. Ingest your data
2. Index and organize it
3. Retrieve relevant information when needed
4. Generate context-aware answers grounded in your knowledge base

The more you feed Hungry AI, the more personalized and useful it becomes.

---

## Configuration Overview

Hungry AI includes a few core settings that control how the system behaves. These settings are important because they define how the AI responds, what it retrieves, and how it organizes information.

You are expected to customize these settings to fit your own use case, data, and workflow.

---

## SYSTEM_PROMPT

This defines the base behavior of the AI.

It controls:

* How the AI responds to commands
* Whether it explains or stays minimal
* Output rules and restrictions

Example:

```python
SYSTEM_PROMPT = """
The assistant is hungry, created by https://github.com/louuuuuu
...
"""
```

If you change this, you change the personality and strictness of the AI.

---

## RAG_MIN_LENGTH

This setting controls when retrieval (RAG) is triggered.

* Short queries below this length will NOT search the database
* Helps avoid unnecessary searches for simple inputs

Example:

```python
RAG_MIN_LENGTH = 10
```

Lower value = more retrieval
Higher value = less retrieval

---

## RAG_SKIP_PHRASES

This is a list of common phrases that skip retrieval completely.

It includes things like:

* greetings (hello, hi)
* small talk (how are you)
* simple responses (thanks, bye)

Example:

```python
RAG_SKIP_PHRASES = {"hi", "hello", "thanks", "bye"}
```

You should edit this list based on how your AI is used.

---

## RULES (Category System)

This defines how incoming data is categorized.

Each rule looks like:

```python
("category", "subcategory", priority, ["keywords"])
```

Example:

```python
("linux", "commands", 10, ["ls", "cd", "cp"])
```

It helps the system:

* Organize knowledge
* Improve search accuracy
* Group related information

You should customize this based on your own data sources and domains.

---

## Why Customization Matters

Hungry AI is not a plug-and-play assistant.

It becomes useful only when you:

* Tune the system prompt for behavior
* Adjust retrieval sensitivity
* Define your own categories and keywords
* Add your own skip rules

The more you customize it, the more it behaves like *your* personal AI system instead of a generic model.

---

## Setup

### Install Dependencies

```bash
pip install -r reqs.txt
```

### PostgreSQL Setup

```bash
sudo -u postgres psql

CREATE DATABASE ragdb;

CREATE USER raguser WITH PASSWORD 'ragpass';

GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;
```

---

## Usage

### Main Application

Run:

```bash
python main.py
```

Available commands:

| Command   | Description                        |
| --------- | ---------------------------------- |
| `/help`   | Show available commands            |
| `/info`   | Display system information         |
| `/clear`  | Clear conversation history         |
| `/list`   | List stored knowledge categories   |
| `/docs`   | Show documentation                 |
| `/status` | Display database and system status |
| `/exit`   | Exit Hungry AI                     |

---

### Data Ingestion

Run:

```bash
python ingest.py --help
```

#### Ingest Web Content

```bash
python ingest.py --mode web --url <URL>
```

#### Ingest Documents

```bash
python ingest.py --mode docs --path <PATH>
```

---

## Ingestion Options

| Option                   | Description                                |
| ------------------------ | ------------------------------------------ |
| `--concurrency <number>` | Number of parallel requests (default: `8`) |
| `--delay <seconds>`      | Delay between requests (default: `0.5`)    |
| `--crawl`                | Follow and crawl discovered links          |
| `--max-pages <number>`   | Maximum pages to crawl                     |

### Example

```bash
python ingest.py \
  --mode web \
  --url https://example.com \
  --crawl \
  --max-pages 50 \
  --concurrency 8
```

---

## Memory Management

### Check Database Status

```bash
python ingest.py --check
```

### Browse Stored Knowledge

List all categories:

```text
list
```

View a specific category:

```text
list <category>
```

Remove empty entries:

```text
list --a
```

### Delete Data

Remove an entire category:

```text
remove <category>
```

Remove a specific record by ID:

```text
remove <category> <id>
```

Delete a subcategory:

```text
delete subcategory <category> <subcategory>
```

---

## Why Hungry AI?

Most AI assistants start every conversation with the same generic knowledge.

Hungry AI is different.

By continuously ingesting and retrieving information from your documents, websites, codebases, notes, and research, it develops a knowledge base unique to you. The result is an AI assistant that understands your work, remembers what matters, and provides answers grounded in your own data.





**Feed it data. It remembers.**

