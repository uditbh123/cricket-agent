import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from knowledge_manager import add_topics_to_db, load_fetched_topics
from crawler import load_discovered_articles, discover_all_cricket_articles, save_discovered_articles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "cricket_db")

def main():
    print("Cricket Agent — Incremental Knowledge Base Builder")
    print("=" * 55)

    # Load articles discovered by crawler
    # If none discovered yet, run discovery automatically
    topics = load_discovered_articles()

    if not topics:
        print("No discovered articles found.")
        print("Running category crawler first...\n")
        topics = discover_all_cricket_articles(
            max_per_category=30,
            max_total=300
        )
        save_discovered_articles(topics)
        print()

    # Show current state
    already_indexed = load_fetched_topics()
    new_topics = [t for t in topics if t.lower() not in already_indexed]

    print(f"Total discovered articles : {len(topics)}")
    print(f"Already in database       : {len(already_indexed)}")
    print(f"New articles to fetch     : {len(new_topics)}")

    if not new_topics:
        print("\nDatabase is up to date. Nothing to do.")
        print("Run crawler.py to discover new articles.")
        return

    print(f"\nFetching {len(new_topics)} new articles...")
    print("-" * 55)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    result = add_topics_to_db(new_topics, vectorstore)

    print("\n" + "=" * 55)
    print(f"✓ Newly added : {len(result['added'])} articles")
    print(f"✗ Failed      : {len(result['failed'])} articles")

    total = load_fetched_topics()
    print(f"\nDatabase now contains: {len(total)} indexed articles")

    if result["failed"]:
        print(f"\n{len(result['failed'])} failed — run ingest.py again to retry")

if __name__ == "__main__":
    main()