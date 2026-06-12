# Hungry AI






## Usage 

# Main.py
- /help
- /info
- /clear
- /list
- /docs
- /status
- /exit



# Ingest.py
- --cocurrency [number] parallel requests (default 8)
- --delay [number] seconds between requests (default 0.5)
- --crawl crawls the web page for more links (1 by default)
- --max-pages increases the crawl pages

- ingest.py --help
- ingest.py --mode web --url [URL]
- ingest.py --mode docs --path [PATH]

__ Memory Check __ 

- ingest.py --check
- list list categories (list [category] to chose the sub category --a to remove ALL empty spaces)
- remove [] deletes category data (delete subcategory [category] [subcategory]) Or remove by id remove [category] [id]
