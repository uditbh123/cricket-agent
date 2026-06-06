import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from agent import get_llm, get_prompt, hybrid_retrieve, ensure_knowledge, format_docs
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: List[Message] = []     # full conversation history from frontend

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

            # Dynamically fetch knowledge for this question
            ensure_knowledge(request.question)

            # Retrieve relevant chunks
            docs = hybrid_retrieve(request.question)
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send sources to frontend first
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # Build conversation history string
            # We only send last 6 messages (3 full turns) to keep
            # the prompt from getting too long
            history_text = ""
            if request.history:
                recent = request.history[-6:]
                lines = []
                for m in recent:
                    label = "User" if m.role == "user" else "Assistant"
                    lines.append(f"{label}: {m.content}")
                history_text = "\nPrevious conversation:\n" + "\n".join(lines) + "\n"

            # Full prompt with memory
            prompt_text = f"""You are a cricket expert assistant with Wikipedia knowledge.

Instructions:
- Use the conversation history to understand context and references
- If the user says "this league", "him", "she", "they" etc. refer to the conversation history to understand who or what they mean
- Answer using the Wikipedia context provided
- Be specific with names, dates, and numbers
- If the answer is not in the context, say so honestly
- Do not invent facts
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