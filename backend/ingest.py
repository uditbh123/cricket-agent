import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Wikipedia API with a proper user agent (required)
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="cricket-agent/1.0 (educational project)"
)

topics = [
    # Core concepts
    "Cricket",
    "Test cricket",
    "One Day International",
    "Twenty20",
    "Laws of cricket",
    "Duckworth-Lewis-Stern method",
    "Cricket statistics",
    "Cricket terminology",
    "Cricket fielding positions",
    "Cricket pitch",
    "LBW (cricket)",

    # Tournaments
    "Indian Premier League",
    "ICC Cricket World Cup",
    "ICC Men's T20 World Cup",
    "The Ashes",
    "ICC Champions Trophy",
    "Border-Gavaskar Trophy",

    # Batting legends
    "Sachin Tendulkar",
    "Virat Kohli",
    "Brian Lara",
    "Ricky Ponting",
    "Kumar Sangakkara",
    "Rohit Sharma",
    "Jacques Kallis",

    # Bowling legends
    "Shane Warne",
    "Muttiah Muralitharan",
    "Glenn McGrath",
    "Wasim Akram",
    "James Anderson (cricketer)",
]

print("Fetching Wikipedia articles...")
all_docs = []

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        # Split into sections for better chunking
        text = page.text
        if len(text) > 100:
            all_docs.append(Document(
                page_content=text,
                metadata={"source": topic, "url": page.fullurl}
            ))
            print(f"  ✓ Fetched: {topic} ({len(text):,} chars)")
        else:
            print(f"  ✗ Too short, skipped: {topic}")
    else:
        print(f"  ✗ Page not found: {topic}")

print(f"\nTotal articles fetched: {len(all_docs)}")

if len(all_docs) == 0:
    print("ERROR: No articles fetched. Check your internet connection.")
    exit(1)

total_chars = sum(len(d.page_content) for d in all_docs)
print(f"Total characters: {total_chars:,}")

# Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(all_docs)
print(f"Total chunks created: {len(chunks)}")

# Load embedding model
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build ChromaDB
print("\nBuilding vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./backend/cricket_db"
)

print(f"\nDone! {len(chunks)} chunks stored in ./backend/cricket_db")