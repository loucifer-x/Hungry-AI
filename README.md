# Hungry AI

Feed it data. It remembers.

Hungry AI is a local AI agent powered by Ollama LLMs that becomes personalized through the information you provide.

Give it your documents, notes, PDFs, websites, codebases, and research. Hungry AI scans, crawls, and remembers what it learns, building a private knowledge base tailored to you.

What it does
- Runs locally with Ollama LLMs
- Scans documents and PDFs
- Crawls websites and online resources
- Builds a personalized knowledge base from your data
- Retrieves relevant information when answering questions
- Keeps your data on your machine

The more you feed Hungry AI, the more it understands your projects, interests, workflows, and knowledge. Instead of relying only on a generic model, it answers using information that matters to you. **Remeber your data is YOUR data.
*A Retrieval-Augmented Generation (RAG) system that searches, retrieves, and reasons over external knowledge sources to generate accurate, context-aware responses. With built in web and document crawlers.*

## Usage 
**Main.py**
- /help
- /info
- /clear
- /list
- /docs
- /status
- /exit

**Ingest.py**
- --cocurrency [number] parallel requests (default 8)
- --delay [number] seconds between requests (default 0.5)
- --crawl crawls the web page for more links (1 by default)
- --max-pages increases the crawl pages

- ingest.py --help
- ingest.py --mode web --url [URL]
- ingest.py --mode docs --path [PATH]

__Memory Check__ 

- ingest.py --check
- list list categories (list [category] to chose the sub category --a to remove ALL empty spaces)
- remove [] deletes category data (delete subcategory [category] [subcategory]) Or remove by id remove [category] [id]
