import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import get_llm, get_prompt, hybrid_retrieve, ensure_knowledge, format_docs
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
    prompt = get_prompt()

    def stream_response():
        try:
            print(f"\nQuestion: {request.question}")

            ensure_knowledge(request.question)

            docs = hybrid_retrieve(request.question)
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            formatted_prompt = prompt.format(
                context=context,
                question=request.question
            )

            for chunk in llm.stream(formatted_prompt):
                # Groq returns AIMessageChunk — extract text content
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield json.dumps({"type": "token", "data": token}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            print(f"Error: {e}")
            yield json.dumps({
                "type": "token",
                "data": f"Something went wrong: {str(e)}"
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream_response(), media_type="text/plain")