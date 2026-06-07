import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from agent import get_llm, hybrid_retrieve, ensure_knowledge, format_docs
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: List[Message] = []

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    llm = get_llm()

    def stream_response():
        try:
            print(f"\nQuestion: {request.question}")
            print(f"History: {len(request.history)} messages")

            # Build history text first so we can pass it to ensure_knowledge
            history_text = ""
            if request.history:
                recent = request.history[-6:]
                lines = []
                for m in recent:
                    label = "User" if m.role == "user" else "Assistant"
                    lines.append(f"{label}: {m.content}")
                history_text = "\n".join(lines)

            # Fetch missing knowledge — passes history so follow-up
            # questions resolve to the right topic
            ensure_knowledge(request.question, history_text)

            # Retrieve relevant chunks from ChromaDB
            docs = hybrid_retrieve(request.question)
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send sources to frontend
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # Build full prompt with conversation history
            prompt_text = f"""You are a cricket expert assistant with Wikipedia knowledge.

Instructions:
- Use the conversation history to understand context and references
- "this league", "him", "they", "it", "the first season" should be resolved using history
- Answer using the Wikipedia context provided
- Be specific with names, dates, and numbers from the context
- Do not invent facts not present in the context

Previous conversation:
{history_text}

Wikipedia Context:
{context}

Current Question: {request.question}

Answer:"""

            # Stream answer token by token
            for chunk in llm.stream(prompt_text):
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