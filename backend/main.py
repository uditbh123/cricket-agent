import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import get_retriever, get_llm, get_prompt, format_docs, rewrite_query
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
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()

    def stream_response():
        try:
            # Step 1 — Rewrite the query for better retrieval
            rewritten = rewrite_query(request.question, llm)
            print(f"Original: {request.question}")
            print(f"Rewritten: {rewritten}")

            # Step 2 — Retrieve using the rewritten query
            source_docs = retriever.invoke(rewritten)
            sources = [doc.page_content for doc in source_docs]
            context = format_docs(source_docs)

            # Send sources to frontend
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # Step 3 — Generate answer using original question
            # (we search with rewritten but answer the original)
            formatted_prompt = prompt.format(
                context=context,
                question=request.question
            )

            # Step 4 — Stream the answer
            for chunk in llm.stream(formatted_prompt):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "token",
                "data": f"Error: {str(e)}"
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )