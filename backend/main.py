import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import get_retriever, get_llm, get_prompt, format_docs
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

    # Get source documents
    source_docs = retriever.invoke(request.question)
    sources = [doc.page_content for doc in source_docs]
    context = format_docs(source_docs)

    # Format the prompt
    formatted_prompt = prompt.format(
        context=context,
        question=request.question
    )

    def stream_response():
        # First send the sources as a JSON line
        yield json.dumps({"type": "sources", "data": sources}) + "\n"

        # Then stream the answer token by token
        for chunk in llm.stream(formatted_prompt):
            yield json.dumps({"type": "token", "data": chunk}) + "\n"

        # Signal completion
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )