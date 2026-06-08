import sys
import os
import json

# ── Path Setup ───────────────────────────────────────────────
# Makes sure Python can find other backend files like knowledge_manager.py
# regardless of where the server is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ── Environment Variables ────────────────────────────────────
# Loads GROQ_API_KEY from backend/.env file
# Never hardcode API keys in source code
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Imports ──────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from knowledge_manager import extract_topics_from_question, add_topics_to_db

# ── Database Path ────────────────────────────────────────────
# Always use absolute path so it works from any working directory
DB_DIR = os.path.join(BASE_DIR, "cricket_db")

# ════════════════════════════════════════════════════════════
# STARTUP — Load Everything Once
# ════════════════════════════════════════════════════════════
# These objects are expensive to create (seconds each).
# We create them ONCE when the server starts and reuse them
# for every request. This is why responses are fast.
# Without caching, every request would take 10+ extra seconds
# just to load models.

print("Loading embedding model...")
# HuggingFace's all-MiniLM-L6-v2 converts text into 384-dimensional vectors.
# Similar meaning = similar vectors = similar numbers.
# This is what makes semantic search work — "little master" finds
# Sachin Tendulkar chunks even without exact keyword match.
# First run downloads ~90MB, subsequent runs load from cache.
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
# ChromaDB is our vector database — stores all Wikipedia chunks
# along with their embeddings for fast similarity search.
# persist_directory means data survives server restarts.
_vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=_embeddings
)

print("Loading LLM (Groq - LLaMA 3.3 70B)...")
# ChatGroq connects to Groq's cloud API.
# llama-3.3-70b-versatile = 70 billion parameters, very capable.
# temperature=0.1 = more focused/deterministic (less random/creative).
# Lower temperature is better for factual Q&A.
# max_tokens=1024 = maximum length of each response.
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)

print("Building BM25 index...")

# ── BM25 Helper Functions ────────────────────────────────────

def _build_bm25() -> BM25Retriever:
    """
    Build a BM25 keyword search index from all documents in ChromaDB.

    BM25 (Best Match 25) is the same algorithm used by search engines
    like Elasticsearch and early Google. It finds documents containing
    the exact keywords from the query, ranked by term frequency.

    Why we need both vector search AND BM25:
    - Vector search: finds semantically similar content
      "who is the little master" → finds Sachin chunks (understands meaning)
    - BM25: finds exact keyword matches
      "Muralitharan 800 wickets" → finds chunks with those exact words

    Combining both gives much better retrieval than either alone.
    """
    raw = _vectorstore.get()
    all_docs = [
        Document(page_content=text)
        for text in raw["documents"]
        if text and len(text.strip()) > 10  # skip empty/tiny chunks
    ]
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = 10  # return top 10 keyword matches
    return retriever


def _refresh_bm25() -> None:
    """
    Rebuild the BM25 index after new documents are added to ChromaDB.

    Called only when ensure_knowledge() adds new Wikipedia articles.
    Without this refresh, newly added articles would be invisible
    to BM25 search (vector search updates automatically, BM25 doesn't).
    """
    global _bm25_retriever
    print("Refreshing BM25 index with new documents...")
    _bm25_retriever = _build_bm25()
    print("BM25 index updated.")


# Build initial BM25 index at startup
_bm25_retriever = _build_bm25()

print("All components ready. Server is accepting requests.")

# ════════════════════════════════════════════════════════════
# PUBLIC API — Functions used by main.py
# ════════════════════════════════════════════════════════════

def get_llm() -> ChatGroq:
    """Return the cached Groq LLM instance."""
    return _llm


def get_vectorstore() -> Chroma:
    """Return the cached ChromaDB vector store."""
    return _vectorstore


def format_docs(docs: list) -> str:
    """
    Convert a list of Document objects into a single formatted string
    that gets sent to the LLM as context.

    Deduplication is important because hybrid search (vector + BM25)
    often returns the same chunk twice. Sending duplicates wastes
    tokens and can confuse the LLM.

    Chunks are separated by '---' so the LLM can distinguish
    where one chunk ends and another begins.
    """
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen and content:
            seen.add(content)
            unique.append(content)
    return "\n\n---\n\n".join(unique)


def rerank_docs(question: str, docs: list) -> list:
    """
    Use the LLM to rerank retrieved chunks by relevance.

    This is a two-stage retrieval pattern used in production RAG:

    Stage 1 — Retrieval (fast, imperfect):
        Embedding model finds 12 roughly relevant chunks in milliseconds.
        Fast but sometimes returns wrong chunks (e.g. Terry Alderman
        when asking about most Test wickets).

    Stage 2 — Reranking (slower, accurate):
        LLM reads all 12 chunks and picks the most relevant ones.
        Much more accurate because LLM truly understands meaning.
        Only adds ~0.5s since Groq 70b runs at 280 tokens/second.

    This fixes retrieval errors that the embedding model can't catch —
    like returning Anderson chunks instead of Muralitharan chunks
    for "most Test wickets" questions.
    """
    if not docs:
        return docs

    # Show only first 300 chars of each chunk to keep reranking prompt short
    # Full content will be sent in the answer prompt — this is just for ranking
    chunks_text = ""
    for i, doc in enumerate(docs):
        chunks_text += f"\nChunk {i+1}:\n{doc.page_content[:300]}\n"

    rerank_prompt = f"""You are evaluating which text chunks are most relevant to answer a cricket question.

Question: {question}

Chunks:
{chunks_text}

Return ONLY a JSON array of chunk numbers ordered by relevance (most relevant first).
Only include chunks that actually help answer the question.
Exclude chunks about unrelated topics.
Example format: [3, 1, 7, 2]

JSON array:"""

    try:
        raw = _llm.invoke(rerank_prompt)
        response = raw.content if hasattr(raw, "content") else str(raw)

        # Strip markdown code blocks if model wraps response
        response = response.replace("```json", "").replace("```", "").strip()

        # Extract the JSON array from response
        start = response.find("[")
        end = response.rfind("]") + 1

        if start != -1 and end > start:
            indices = json.loads(response[start:end])

            # Reorder docs according to LLM ranking
            reranked = []
            seen_indices = set()
            for idx in indices:
                # Validate index is a real chunk number
                if isinstance(idx, int) and 1 <= idx <= len(docs):
                    if idx not in seen_indices:
                        reranked.append(docs[idx - 1])
                        seen_indices.add(idx)

            # Append any remaining docs not included in LLM ranking
            # (safety net — ensures we don't lose relevant chunks)
            for i, doc in enumerate(docs):
                if (i + 1) not in seen_indices:
                    reranked.append(doc)

            print(f"Reranked: {len(reranked)} chunks returned")
            # Return top 8 after reranking — higher quality than top 12 unranked
            return reranked[:8]

    except Exception as e:
        print(f"Reranking failed, using original order: {e}")

    # If reranking fails for any reason, return original docs unchanged
    return docs


def hybrid_retrieve(question: str, history_text: str = "") -> list:
    """
    Retrieve relevant chunks using both vector search and BM25.

    Two-retriever approach:
    1. Vector retriever — semantic similarity via ChromaDB
       Converts question to embedding, finds closest chunk embeddings
       Good for: conceptual questions, nicknames, paraphrased queries

    2. BM25 retriever — keyword matching
       Finds chunks containing the exact words from the question
       Good for: proper names, statistics, specific terminology

    Results from both are combined and deduplicated, giving us
    the benefits of both approaches.

    history_text parameter:
    Used for follow-up questions. If user asks "who won the first season?"
    after discussing Nepal Premier League, we use the history to build
    a better search query so we search for NPL, not IPL.
    """
    # Use last user message from history as context for vague follow-ups
    # But keep it short — too much history makes retrieval noisy
    if history_text:
        # Take only the last 2 lines of history to avoid noise
        last_lines = history_text.strip().split("\n")[-2:]
        context = " ".join(last_lines)
        search_query = f"{context} {question}".strip()
    else:
        search_query = question

    print(f"Retrieval query: {search_query[:100]}")

    # Vector search — semantic similarity
    vector_retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 10}  # fetch top 10 semantically similar chunks
    )
    vector_docs = vector_retriever.invoke(search_query)

    # BM25 search — keyword matching
    bm25_docs = _bm25_retriever.invoke(search_query)

    # Combine results, deduplicate, preserve order (vector first)
    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            combined.append(doc)

    print(f"Retrieved {len(combined)} unique chunks before reranking")

    # Return top 12 — rerank_docs will trim this to top 8 most relevant
    return combined[:12]


def ensure_knowledge(question: str, history_text: str = "") -> None:
    """
    Dynamically fetch Wikipedia knowledge for topics in the question.

    This is what makes the knowledge base self-maintaining:
    - First time: "Who is Jasprit Bumrah?" → fetches Wikipedia → adds to DB
    - Second time: already in DB → skips fetch instantly
    - BM25 index is rebuilt only when new docs are actually added

    history_text helps resolve follow-up questions correctly:
    "who won the first season?" after NPL discussion
    → LLM extracts "Nepal Premier League", not "Indian Premier League"

    The function is intentionally fault-tolerant — if Wikipedia fetch
    fails, the agent still answers from existing knowledge.
    """
    global _bm25_retriever

    # Ask LLM to extract Wikipedia search topics from the question
    # This handles nicknames, slang, vague references — anything
    topics = extract_topics_from_question(question, _llm, history_text)

    if not topics:
        print("No new topics to fetch")
        return

    print(f"Topics to fetch: {topics}")
    result = add_topics_to_db(topics, _vectorstore)

    if result["added"]:
        print(f"Newly added to DB: {result['added']}")
        # Rebuild BM25 so new articles are searchable via keyword matching
        # Vector search updates automatically — BM25 requires manual rebuild
        _refresh_bm25()

    if result["skipped"]:
        print(f"Already in DB: {result['skipped']}")

    if result["failed"]:
        print(f"Failed to fetch (will retry next time): {result['failed']}")