import wikipedia 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 

topics = [
    "Cricket",
    "Test criket",
    "One Day International",
    "Twenty20",
    "Indian Premier League",
    "Laws of cricket",
    "Sachin Tendulkar",
    "Virat Kohli",
    "Brian Lara",
    "Shane Warne",
    "ICC Cricket World Cup",
    "The Ashes",
    "Duckworth-Lewis-Stern method",
]

print("Fetching Wikipedia articles...")
all_text = []

for topic in topics:
    try:
        page = wikipedia.page(topic)
        all_text.append(page.content)
        print(f" Fetched: {topic}")
    except Exception as e:
        print(f" Skipped {topic}: {e}")

combined_text ="\n\n".join(all_text)
print(f"\nTotal characters fetched: {len(combined_text)}")

print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.create_documents([combined_text])
print(f"Total chunks: {len(chunks)}")

print("\nLoading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("\nBuilding vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./cricket_db"

)
