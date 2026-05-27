import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory="./backend/cricket_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_llm():
    return OllamaLLM(model="llama3.2:1b")

def get_prompt():
    prompt_template = """You are a cricket expert assistant. Use the following context from Wikipedia to answer the question. Be clear and concise. If the answer is not in the context, say "I don't have enough information about that in my cricket knowledge base."

Context:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)