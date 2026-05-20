import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./backend/cricket_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(model="llama3.2")

prompt_template = """You are a cricket expert assistant. Use the following context from Wikipedia to answer the question. If the answer is not in the context, say "I don't have enough information about that in my cricket knowledge base."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question: str):
    source_docs = retriever.invoke(question)
    sources = [doc.page_content for doc in source_docs]
    answer = rag_chain.invoke(question)
    return answer, sources