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
    # ── Core Cricket Knowledge ───────────────────────────────
    "Cricket",
    "Cricket terminology",
    "Cricket laws and regulations",
    "Cricket statistics",
    "Formats of cricket",
    "History of cricket",
    "Cricket culture",

    # ── Players by Nationality ───────────────────────────────
    # Major Test nations
    "Indian cricketers",
    "Australian cricketers",
    "English cricketers",
    "Pakistani cricketers",
    "Sri Lankan cricketers",
    "West Indian cricketers",
    "South African cricketers",
    "New Zealand cricketers",
    "Bangladeshi cricketers",
    "Zimbabwean cricketers",

    # Associate nations — growing cricket markets
    "Afghan cricketers",
    "Irish cricketers",
    "Nepali cricketers",
    "Scottish cricketers",
    "Netherlands cricketers",
    "UAE cricketers",
    "Namibian cricketers",
    "Oman cricketers",

    # ── Tournaments and Competitions ─────────────────────────
    # ICC events
    "Cricket World Cup",
    "ICC Cricket World Cup",
    "ICC Men's T20 World Cup",
    "ICC Champions Trophy",
    "ICC World Test Championship",
    "ICC Women's Cricket World Cup",
    "ICC Under-19 Cricket World Cup",

    # Bilateral series
    "The Ashes series",
    "Border-Gavaskar Trophy",
    "Test cricket",
    "One Day International cricket",
    "Twenty20 International cricket",

    # Domestic T20 leagues — all major ones
    "Indian Premier League seasons",
    "Big Bash League seasons",
    "Pakistan Super League seasons",
    "Caribbean Premier League seasons",
    "Bangladesh Premier League seasons",
    "Lanka Premier League seasons",
    "SA20 seasons",
    "The Hundred seasons",
    "Nepal Premier League seasons",

    # ── Records ──────────────────────────────────────────────
    "Cricket records",
    "Cricket World Cup records",
    "Test cricket records",
    "One Day International cricket records",
    "Twenty20 International records",
    "Indian Premier League records",

    # ── Grounds and Venues ───────────────────────────────────
    "Cricket grounds",
    "Cricket grounds in India",
    "Cricket grounds in England",
    "Cricket grounds in Australia",
    "Cricket grounds in Pakistan",
    "Cricket grounds in South Africa",
    "Cricket grounds in the West Indies",
    "Cricket grounds in New Zealand",
    "Cricket grounds in Sri Lanka",
    "Cricket grounds in Bangladesh",

    # ── Governing Bodies and Officials ───────────────────────
    "Cricket governing bodies",
    "Cricket umpires",
    "Cricket coaches",

    # ── Women's Cricket ───────────────────────────────────────
    "Women's cricket",
    "Women cricketers",
    "Indian women cricketers",
    "Australian women cricketers",
    "English women cricketers",

    # ── Technical Aspects ─────────────────────────────────────
    "Cricket batting",
    "Cricket bowling",
    "Cricket fielding positions",
    "Cricket equipment",
    "Cricket clothing and equipment",

    # ── Nepal Cricket ─────────────────────────────────────────
    "Cricket in Nepal",
    "Nepal Premier League seasons",

    # ── Player Categories by Skill ────────────────────────────
    "Cricket captains",
    "Cricket all-rounders",
    "Wicket-keepers",
    "Left-handed batsmen",
    "Spin bowlers",
    "Fast bowlers",

    # ── Historical and Cultural ───────────────────────────────
    "History of cricket",
    "Cricket in popular culture",
    "Cricket writers and broadcasters",
    "Cricket controversies",
    "Tied cricket matches",
    "Super Overs in cricket",
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
        max_per_category=50,
        max_total=1000
    )

    print(f"\nDiscovered {len(articles)} unique cricket articles")
    save_discovered_articles(articles)

    print("\nSample of discovered articles:")
    for article in articles[:20]:
        print(f"  - {article}")

    print(f"\nNow run: python backend/ingest.py")
    print("ingest.py will automatically use these discovered articles.")