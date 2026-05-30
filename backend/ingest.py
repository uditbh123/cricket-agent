import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from knowledge_manager import add_topics_to_db

SEED_TOPICS = [
    "Cricket",
    "Test cricket",
    "One Day International cricket",
    "Twenty20 cricket",
    "Laws of cricket",
    "Indian Premier League",
    "ICC Cricket World Cup",
    "Sachin Tendulkar",
    "Virat Kohli",
    "Shane Warne",
    "Muttiah Muralitharan",
    "Brian Lara",
    "Rohit Sharma",
    "MS Dhoni",
    "Jasprit Bumrah",
    "Duckworth Lewis Stern method cricket",
    "Cricket fielding positions",
    "Cricket batting",
    "Cricket bowling",
]

print("Setting up Cricket Agent knowledge base...")
print("=" * 50)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./backend/cricket_db",
    embedding_function=embeddings
)

print(f"\nSeeding {len(SEED_TOPICS)} base topics...")
result = add_topics_to_db(SEED_TOPICS, vectorstore)

print("\n" + "=" * 50)
print(f"✓ Added:  {len(result['added'])} topics")
print(f"✗ Failed: {len(result['failed'])} topics")
if result["failed"]:
    print(f"  Failed topics: {result['failed']}")
print("\nKnowledge base ready!")