import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import (
    get_retriever, get_llm, get_prompt, get_vectorstore,
    format_docs, hybrid_retrieve, ensure_knowledge
)
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    llm = get_llm()
    vectorstore = get_vectorstore()
    prompt = get_prompt()

    def stream_response():
        try:
            print(f"\nQuestion: {request.question}")

            # Dynamically fetch Wikipedia knowledge for this question
            # Works for any question — no hardcoding needed
            ensure_knowledge(request.question, llm, vectorstore)

            # Fresh retriever — includes any newly added docs
            vector_retriever, bm25_retriever = get_retriever()

            # Hybrid retrieval
            docs = hybrid_retrieve(
                request.question,
                vector_retriever,
                bm25_retriever
            )

            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # Stream the answer
            formatted_prompt = prompt.format(
                context=context,
                question=request.question
            )

            for chunk in llm.stream(formatted_prompt):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            print(f"Error: {e}")
            yield json.dumps({
                "type": "token",
                "data": f"Something went wrong: {str(e)}"
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream_response(), media_type="text/plain")