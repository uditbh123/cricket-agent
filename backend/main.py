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
from agent import get_llm, hybrid_retrieve, ensure_knowledge, format_docs, rerank_docs

# ── Create the FastAPI application ──────────────────────────
app = FastAPI()

# ── Allow React frontend (localhost:5173) to talk to this API
# Without this, the browser blocks cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data models ──────────────────────────────────────────────
# Pydantic models validate incoming JSON automatically
# If the request doesn't match, FastAPI returns a 422 error

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str    # the message text

class QuestionRequest(BaseModel):
    question: str               # current question from user
    history: List[Message] = [] # full conversation so far (default empty)

# ── Health check endpoint ────────────────────────────────────
# React can ping this to verify the backend is running
@app.get("/health")
def health():
    return {"status": "running"}

# ── Main endpoint — receives question, streams answer ────────
@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Get the cached LLM (loaded once at startup in agent.py)
    llm = get_llm()

    def stream_response():
        """
        Generator function that yields JSON lines one at a time.
        React reads these as they arrive, showing words as they stream.
        Three event types:
          {"type": "sources", "data": [...]}  — sent first, the retrieved chunks
          {"type": "token",   "data": "word"} — each word of the answer
          {"type": "done"}                    — signals stream is complete
        """
        try:
            print(f"\nQuestion: {request.question}")
            print(f"History: {len(request.history)} messages")

            # ── Step 1: Build conversation history string ────
            # We take the last 6 messages (3 full turns) to keep
            # the prompt from getting too large for the LLM context window
            history_text = ""
            if request.history:
                recent = request.history[-12:]
                lines = []
                for m in recent:
                    label = "User" if m.role == "user" else "Assistant"
                    lines.append(f"{label}: {m.content}")
                history_text = "\n".join(lines)

            # ── Step 2: Dynamically fetch missing knowledge ──
            # If user asks about a topic not in our DB, this fetches
            # the Wikipedia article and adds it to ChromaDB
            # We pass history_text so follow-up questions like
            # "who won the first season?" resolve to the right topic
            ensure_knowledge(request.question, history_text)

            # ── Step 3: Hybrid retrieval ─────────────────────
            # Searches ChromaDB using both vector similarity and BM25
            # history_text is used to expand vague follow-up questions
            docs = hybrid_retrieve(request.question, history_text)
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send retrieved sources to frontend first
            # User can expand these to see where the answer came from
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # ── Step 4: Build prompt with history + context ──
            # The LLM sees: conversation history + Wikipedia context + question
            # This is what makes memory and grounded answers work together
            prompt_text = f"""You are a cricket expert assistant.

STRICT RULES:
1. Answer ONLY using facts from your knowledge base
2. NEVER mention "context", "Wikipedia", "sources" or "provided information" in your answer
3. NEVER explain where your information comes from or does not come from
4. If you don't have the answer, say ONLY: "I don't have information about that in my knowledge base."
5. Never say things like "based on the context" or "the context primarily focuses on"
6. NEVER invent facts, names, statistics, or dates
7. Use conversation history to understand references like "this league", "him", "they", "it"

Previous conversation:
{history_text}

Knowledge base:
{context}

Current Question: {request.question}

Answer:"""

            # ── Step 5: Stream the answer token by token ─────
            # Groq returns AIMessageChunk objects, not plain strings
            # We extract .content from each chunk and send to frontend
            for chunk in llm.stream(prompt_text):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield json.dumps({"type": "token", "data": token}) + "\n"

            # Signal to frontend that streaming is complete
            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            print(f"Error: {e}")
            # Send error as a token so frontend displays it in the chat
            yield json.dumps({
                "type": "token",
                "data": f"Something went wrong: {str(e)}"
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    # StreamingResponse sends the generator output incrementally
    # media_type="text/plain" so frontend can read it as a stream
    return StreamingResponse(stream_response(), media_type="text/plain")