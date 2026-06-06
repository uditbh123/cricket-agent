import sys
import os
import requests
import time
import json
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Categories to crawl — Wikipedia's own cricket taxonomy
# Adding a category here pulls ALL articles inside it automatically
CRICKET_CATEGORIES = [
    "Cricket",
    "Cricket terminology",
    "Cricket laws and regulations",
    "Cricket statistics",
    "Formats of cricket",
    "Indian cricketers",
    "Australian cricketers",
    "English cricketers",
    "Pakistani cricketers",
    "Sri Lankan cricketers",
    "West Indian cricketers",
    "South African cricketers",
    "New Zealand cricketers",
    "Bangladeshi cricketers",
    "Cricket competitions",
    "Cricket World Cup",
    "Indian Premier League seasons",
    "Test cricket",
    "One Day International cricket",
    "Twenty20 International cricket",
    "Cricket grounds",
    "Cricket records",
    "Nepali cricketers",
    "Nepal Premier League seasons",
    "Cricket governing bodies",        # ← ICC, BCCI, ECB etc
    "Cricket stadiums",                # ← all major grounds
    "Cricket World Cup tournaments",   # ← every World Cup
    "Women cricketers",                # ← women's cricket
    "Cricket umpires",                 # ← umpires and officials
    "Dismissed cricket players",       # ← retired legends

]

HEADERS = {
    "User-Agent": "cricket-agent/1.0 (educational project)"
}

def get_articles_in_category(category: str, max_articles: int = 50) -> list[str]:
    """
    Get all article titles in a Wikipedia category.
    Returns a list of article titles — not subcategories.
    """
    url = "https://en.wikipedia.org/w/api.php"
    articles = []
    continue_param = {}

    while len(articles) < max_articles:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "page",  # only articles, not subcategories
            "cmlimit": 50,
            "format": "json",
            **continue_param
        }

        try:
            time.sleep(1)
            response = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if not response.text.strip():
                break

            data = response.json()
            members = data.get("query", {}).get("categorymembers", [])
            articles.extend([m["title"] for m in members])

            # Handle pagination
            if "continue" in data:
                continue_param = data["continue"]
            else:
                break

        except Exception as e:
            print(f"  Error fetching category '{category}': {e}")
            break

    return articles[:max_articles]

def get_subcategories(category: str) -> list[str]:
    """Get subcategory names inside a category."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": 20,
        "format": "json",
    }

    try:
        time.sleep(1)
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if not response.text.strip():
            return []

        data = response.json()
        members = data.get("query", {}).get("categorymembers", [])
        # Strip "Category:" prefix
        return [m["title"].replace("Category:", "") for m in members]

    except Exception:
        return []

def discover_all_cricket_articles(
    max_per_category: int = 30,
    max_total: int = 500
) -> list[str]:
    """
    Discover all cricket articles by crawling Wikipedia categories.
    Returns a deduplicated list of article titles.
    """
    all_articles = set()
    processed_categories = set()

    # Queue starts with our seed categories
    category_queue = list(CRICKET_CATEGORIES)

    print(f"Starting discovery from {len(category_queue)} seed categories...")
    print(f"Target: up to {max_total} unique articles")
    print("-" * 50)

    while category_queue and len(all_articles) < max_total:
        category = category_queue.pop(0)

        if category in processed_categories:
            continue

        processed_categories.add(category)
        print(f"Scanning: {category}...", end=" ")

        # Get articles in this category
        articles = get_articles_in_category(category, max_per_category)
        new_articles = [a for a in articles if a not in all_articles]
        all_articles.update(new_articles)

        print(f"{len(new_articles)} new articles (total: {len(all_articles)})")

        if len(all_articles) >= max_total:
            break

    return list(all_articles)

def save_discovered_articles(articles: list[str]):
    """Save discovered articles to a file for ingest.py to use."""
    path = os.path.join(BASE_DIR, "discovered_articles.json")
    with open(path, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"\nSaved {len(articles)} articles to backend/discovered_articles.json")
    return path

def load_discovered_articles() -> list[str]:
    """Load previously discovered articles."""
    path = os.path.join(BASE_DIR, "discovered_articles.json")
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    print("Cricket Wikipedia Category Crawler")
    print("=" * 50)
    print("This discovers ALL cricket articles automatically")
    print("from Wikipedia's category system.")
    print("No manual topic listing needed.\n")

    articles = discover_all_cricket_articles(
        max_per_category=30,
        max_total=300
    )

    print(f"\nDiscovered {len(articles)} unique cricket articles")
    save_discovered_articles(articles)

    print("\nSample of discovered articles:")
    for article in articles[:20]:
        print(f"  - {article}")

    print(f"\nNow run: python backend/ingest.py")
    print("ingest.py will automatically use these discovered articles.")