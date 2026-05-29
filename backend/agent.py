import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── Embeddings ──────────────────────────────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Load all docs from ChromaDB for BM25 ────────────────────
def load_all_docs():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory="./backend/cricket_db",
        embedding_function=embeddings
    )
    # Pull every stored chunk out of ChromaDB
    raw = vectorstore.get()
    docs = [
        Document(page_content=text)
        for text in raw["documents"]
    ]
    return docs, vectorstore

# ── Hybrid Retriever (Vector + BM25) ────────────────────────
def get_retriever():
    docs, vectorstore = load_all_docs()

    # Vector retriever — finds semantically similar chunks
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    # BM25 retriever — finds exact keyword matches
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 8

    # Ensemble combines both — 50% vector, 50% keyword
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever

# ── LLM ─────────────────────────────────────────────────────
def get_llm():
    return OllamaLLM(
        model="llama3.2:1b",
        temperature=0.1  # lower = more focused, less random
    )

# ── Prompt ──────────────────────────────────────────────────
def get_prompt():
    prompt_template = """You are a cricket expert assistant with deep knowledge of the game.

Use ONLY the following context from Wikipedia to answer the question.
Be specific, accurate, and cite specific facts from the context.
If the context does not contain enough information, say exactly:
"I don't have enough information about that in my cricket knowledge base."
Do NOT make up statistics, names, or dates.

Context:
{context}

Question: {question}

Answer (be specific and use facts from the context):"""

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

# ── Format docs ─────────────────────────────────────────────
def format_docs(docs):
    # Deduplicate chunks (hybrid search can return duplicates)
    seen = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(content)

    return "\n\n---\n\n".join(unique_docs)