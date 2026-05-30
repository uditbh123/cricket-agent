import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from knowledge_manager import add_topics_to_db, load_fetched_topics

# ── Master topic list ─────────────────────────────────────────
# Add new topics here anytime — already-indexed ones are skipped
# Never remove topics from this list
TOPICS = [
    # Formats
    "Cricket",
    "Test cricket",
    "One Day International cricket",
    "Twenty20 cricket",
    "The Hundred cricket",

    # Rules and concepts
    "Laws of Cricket",
    "Duckworth-Lewis-Stern method",
    "Cricket fielding positions",
    "Batting cricket",
    "Bowling cricket",
    "Cricket terminology",
    "LBW cricket law",
    "No-ball cricket",
    "Wide cricket",
    "Cricket pitch",
    "Cricket statistics",

    # Tournaments
    "Indian Premier League",
    "ICC Cricket World Cup",
    "ICC Men's T20 World Cup",
    "ICC Champions Trophy",
    "The Ashes",
    "Border-Gavaskar Trophy",
    "Big Bash League",
    "Caribbean Premier League",
    "Pakistan Super League",

    # Batters
    "Sachin Tendulkar",
    "Virat Kohli",
    "Brian Lara",
    "Ricky Ponting",
    "Kumar Sangakkara",
    "Rohit Sharma",
    "Jacques Kallis",
    "MS Dhoni",
    "AB de Villiers",
    "Steve Smith cricketer",
    "Kane Williamson",
    "Joe Root",
    "Babar Azam",
    "David Warner cricketer",

    # Bowlers
    "Shane Warne",
    "Muttiah Muralitharan",
    "Glenn McGrath",
    "Wasim Akram",
    "James Anderson cricketer",
    "Jasprit Bumrah",
    "Shoaib Akhtar",
    "Brett Lee",
    "Stuart Broad",
    "Anil Kumble",
    "Ravichandran Ashwin",

    # All rounders
    "Imran Khan cricketer",
    "Kapil Dev",
    "Richard Hadlee",
    "Ian Botham",
    "Andrew Flintoff",
    "Ben Stokes",
]

def main():
    print("Cricket Agent — Incremental Knowledge Base Builder")
    print("=" * 55)

    # Show current state before doing anything
    already_indexed = load_fetched_topics()
    new_topics = [t for t in TOPICS if t.lower() not in already_indexed]
    
    print(f"Total topics in master list : {len(TOPICS)}")
    print(f"Already in database         : {len(already_indexed)}")
    print(f"New topics to fetch         : {len(new_topics)}")

    if not new_topics:
        print("\nDatabase is already up to date. Nothing to do.")
        print("To add more topics, add them to the TOPICS list in ingest.py")
        return

    print(f"\nFetching {len(new_topics)} new topics...")
    print("(Already-indexed topics are skipped automatically)")
    print("-" * 55)

    # Connect to existing DB — never wipes it
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./backend/cricket_db",
        embedding_function=embeddings
    )

    result = add_topics_to_db(new_topics, vectorstore)

    print("\n" + "=" * 55)
    print(f"✓ Newly added : {len(result['added'])} topics")
    print(f"✗ Failed      : {len(result['failed'])} topics")

    if result["added"]:
        print(f"\nAdded: {result['added']}")

    if result["failed"]:
        print(f"\nFailed (will retry next run): {result['failed']}")
        print("Run ingest.py again to retry failed topics.")

    total_indexed = load_fetched_topics()
    print(f"\nDatabase now contains: {len(total_indexed)} indexed topics")
    print("Run ingest.py anytime to add new topics safely.")

if __name__ == "__main__":
    main()