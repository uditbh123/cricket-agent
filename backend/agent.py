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

_bm25_retriever = _build_bm25()

print("All components ready.")

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

def hybrid_retrieve(question: str):
    vector_retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = _bm25_retriever.invoke(question)

    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            combined.append(doc)

    return combined[:12]

def ensure_knowledge(question: str):
    global _bm25_retriever

    topics = extract_topics_from_question(question, _llm)
    if not topics:
        return

    print(f"Topics to fetch: {topics}")
    result = add_topics_to_db(topics, _vectorstore)

    if result["added"]:
        print(f"Newly added: {result['added']}")
        # Rebuild BM25 only when new docs were added
        _bm25_retriever = _build_bm25()

    if result["skipped"]:
        print(f"Already in DB: {result['skipped']}")