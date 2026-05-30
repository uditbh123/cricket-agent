import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from knowledge_manager import extract_topics_from_question, add_topics_to_db

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(
        persist_directory="./backend/cricket_db",
        embedding_function=get_embeddings()
    )

def get_retriever():
    vectorstore = get_vectorstore()
    raw = vectorstore.get()

    all_docs = [
        Document(page_content=text)
        for text in raw["documents"]
        if text and len(text.strip()) > 10
    ]

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    return vector_retriever, bm25_retriever

def get_llm():
    return OllamaLLM(model="llama3.2:1b", temperature=0.1)

def get_prompt():
    template = """You are a cricket expert assistant with access to Wikipedia knowledge.

Your job:
- Answer any cricket-related question using the context provided
- If the question uses a nickname, slang, or informal phrasing, figure out who or what they mean from the context
- Be specific — use real names, dates, and numbers from the context
- If the context genuinely doesn't have the answer, say so honestly

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

def hybrid_retrieve(question: str, vector_retriever, bm25_retriever):
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    seen = set()
    combined = []
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            combined.append(doc)

    return combined[:12]

def ensure_knowledge(question: str, llm, vectorstore):
    """
    Extract topics from the question and fetch their
    Wikipedia articles if not already in the database.
    Works for any question — nicknames, slang, vague phrasing,
    anything. The LLM figures out what to search for.
    """
    topics = extract_topics_from_question(question, llm)
    if topics:
        print(f"Topics to fetch: {topics}")
        result = add_topics_to_db(topics, vectorstore)
        if result["added"]:
            print(f"Newly added to DB: {result['added']}")
        if result["skipped"]:
            print(f"Already in DB: {result['skipped']}")
    return topics