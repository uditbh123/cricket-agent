import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from knowledge_manager import extract_topics_from_question, add_topics_to_db

DB_DIR = os.path.join(BASE_DIR, "cricket_db")

# Load everything ONCE at startup 

print("Loading embedding model...")
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
_vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=_embeddings
)

print("Loading LLM (Groq)...")
_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024,
)

print("Building BM25 index...")

def _build_bm25():
    raw = _vectorstore.get()
    all_docs = [
        Document(page_content=text)
        for text in raw["documents"]
        if text and len(text.strip()) > 10
    ]
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = 10
    return retriever


def _refresh_bm25():
    """Rebuilds BM25 index after new docs are added to ChromaDB."""
    global _bm25_retriever
    print("Refreshing BM25 index...")
    _bm25_retriever = _build_bm25()
    print("BM25 index updated.")


_bm25_retriever = _build_bm25()

# Public API 

def get_llm():
    return _llm

def get_vectorstore():
    return _vectorstore

def get_prompt():
    template = """You are a cricket expert assistant with Wikipedia knowledge.

Your job:
- Answer any cricket question using the context provided
- Be specific — use real names, dates, and numbers from the context
- If the context genuinely does not contain the answer, say so honestly
- Do NOT invent statistics, names, or dates not in the context
- For future events not in the context, say you don't have that information

Context:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def format_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen and content:
            seen.add(content)
            unique.append(content)
    return "\n\n---\n\n".join(unique)

def hybrid_retrieve(question: str, context_hint: str = ""):
    """
    Retrieve relevant chunks. If context_hint is provided,
    it's combined with the question for better retrieval
    on vague follow-up questions.
    """
    # Use expanded query if we have context
    search_query = f"{context_hint} {question}".strip() if context_hint else question

    vector_retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    vector_docs = vector_retriever.invoke(search_query)
    bm25_docs = _bm25_retriever.invoke(search_query)

    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            combined.append(doc)

    return combined[:12]

def ensure_knowledge(question: str, history_text: str = ""):
    """
    Fetch Wikipedia knowledge for topics in the question.
    Uses conversation history so follow-up questions like
    'who won the first season?' resolve correctly to the
    topic being discussed, not a random guess.
    """
    global _bm25_retriever

    topics = extract_topics_from_question(question, _llm, history_text)
    if not topics:
        return

    print(f"Topics to fetch: {topics}")
    result = add_topics_to_db(topics, _vectorstore)

    if result["added"]:
        print(f"Newly added: {result['added']}")
        _refresh_bm25()

    if result["skipped"]:
        print(f"Already in DB: {result['skipped']}")