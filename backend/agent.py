import sys
import os

# Make sure Python can find other backend files (knowledge_manager etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Load GROQ_API_KEY from backend/.env file
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from knowledge_manager import extract_topics_from_question, add_topics_to_db

# Absolute path to ChromaDB — works regardless of where you run the server from
DB_DIR = os.path.join(BASE_DIR, "cricket_db")

# ── Load everything ONCE at server startup ───────────────────
# These lines run when uvicorn first imports this module
# Every request after that reuses these cached objects
# This is why requests are fast — no reloading on each call

print("Loading embedding model...")
# Converts text into 384-dimensional vectors representing meaning
# Used for semantic similarity search in ChromaDB
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
# Connect to the local vector database built by ingest.py
# persist_directory tells ChromaDB where the database files are stored
_vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=_embeddings
)

print("Loading LLM (Groq)...")
# ChatGroq connects to Groq's cloud API
# llama-3.1-8b-instant is fast (300+ tokens/sec) and free tier available
# temperature=0.1 means more focused/deterministic answers (less random)
_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024,
)

print("Building BM25 index...")

def _build_bm25():
    """
    Build a BM25 keyword search index from all documents in ChromaDB.
    BM25 is the same algorithm Google uses for keyword matching.
    It complements vector search by finding exact keyword matches
    that semantic search sometimes misses.
    """
    raw = _vectorstore.get()
    all_docs = [
        Document(page_content=text)
        for text in raw["documents"]
        if text and len(text.strip()) > 10
    ]
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = 10  # return top 10 matches
    return retriever

def _refresh_bm25():
    """
    Rebuild the BM25 index after new documents are added to ChromaDB.
    Called only when ensure_knowledge adds new Wikipedia articles.
    Without this, newly added articles wouldn't appear in BM25 results.
    """
    global _bm25_retriever
    print("Refreshing BM25 index...")
    _bm25_retriever = _build_bm25()
    print("BM25 index updated.")

# Build the initial BM25 index at startup
_bm25_retriever = _build_bm25()

print("All components ready.")

# ── Public functions used by main.py ─────────────────────────

def get_llm():
    """Return the cached Groq LLM instance."""
    return _llm

def get_vectorstore():
    """Return the cached ChromaDB vector store."""
    return _vectorstore

def format_docs(docs):
    """
    Convert a list of Document objects into a single string.
    Deduplicates chunks so the same text isn't sent twice.
    Chunks are separated by --- so the LLM can distinguish them.
    """
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen and content:
            seen.add(content)
            unique.append(content)
    return "\n\n---\n\n".join(unique)

def hybrid_retrieve(question: str, history_text: str = ""):
    """
    Retrieve relevant chunks using both vector search and BM25.

    Vector search: finds chunks that are semantically similar
    BM25 search: finds chunks that contain matching keywords

    Combining both gives better results than either alone.
    For example:
    - "who is the little master" → vector finds Sachin Tendulkar chunks
    - "how many centuries" → BM25 finds chunks with the word "centuries"

    history_text is used to expand vague follow-up questions.
    "who won the first season?" + "Nepal Premier League" context
    → searches for "Nepal Premier League who won the first season?"
    """
    # For follow-up questions, combine with last user message for context
    search_query = f"{history_text} {question}".strip() if history_text else question

    # Vector retriever — semantic similarity
    vector_retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    vector_docs = vector_retriever.invoke(search_query)

    # BM25 retriever — keyword matching
    bm25_docs = _bm25_retriever.invoke(search_query)

    # Combine results and remove duplicates
    # Vector docs come first — they tend to be more relevant
    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            combined.append(doc)

    # Return top 12 unique chunks
    return combined[:12]

def ensure_knowledge(question: str, history_text: str = ""):
    """
    Check if we have Wikipedia knowledge for this question.
    If not, fetch it automatically and add to the database.

    This is what makes the knowledge base grow with usage.
    First time someone asks about Jasprit Bumrah → fetches his Wikipedia page
    Second time → already in DB, skips fetch instantly

    history_text helps resolve follow-up questions:
    "who won the first season?" after discussing NPL
    → extracts "Nepal Premier League" as the topic, not IPL
    """
    global _bm25_retriever

    # Ask LLM to extract Wikipedia search topics from the question
    topics = extract_topics_from_question(question, _llm, history_text)
    if not topics:
        return

    print(f"Topics to fetch: {topics}")
    result = add_topics_to_db(topics, _vectorstore)

    if result["added"]:
        print(f"Newly added: {result['added']}")
        # Rebuild BM25 so newly added articles are searchable
        _refresh_bm25()

    if result["skipped"]:
        print(f"Already in DB: {result['skipped']}")