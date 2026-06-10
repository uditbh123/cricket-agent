import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from agent import get_llm, hybrid_retrieve, ensure_knowledge, format_docs, rerank_docs, validate_context
import json

# ── Create FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="Cricket AI Agent API",
    description="RAG-powered cricket Q&A with Wikipedia knowledge base",
    version="1.0.0"
)

# ── CORS Middleware ──────────────────────────────────────────
# Without this, browsers block requests from React (port 5173)
# to FastAPI (port 8000) — a security feature called Same-Origin Policy
# allow_origins=["*"] allows all origins — fine for development
# In production, replace * with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Models ──────────────────────────────────────────────
# Pydantic models do two things:
# 1. Validate incoming JSON automatically (wrong type = 422 error)
# 2. Document the API schema (visible at /docs)

class Message(BaseModel):
    """A single message in the conversation history."""
    role: str       # "user" or "assistant"
    content: str    # the actual message text

class QuestionRequest(BaseModel):
    """
    The request body for the /ask endpoint.
    Frontend sends this every time user submits a question.
    """
    question: str                   # current question from user
    history: List[Message] = []     # full conversation so far (empty by default)

# ── Helper: Build History Text ───────────────────────────────
def build_history_text(history: List[Message], max_messages: int = 12) -> str:
    """
    Convert conversation history list into a formatted string
    for the LLM prompt.

    We limit to last 12 messages (6 turns) to keep the prompt
    from getting too long. Beyond that, context gets noisy and
    the LLM loses focus on the current question.

    Format:
        User: what is the Nepal Premier League?
        Assistant: The Nepal Premier League is...
        User: who won the first season?
    """
    if not history:
        return ""

    recent = history[-max_messages:]
    lines = []
    for m in recent:
        label = "User" if m.role == "user" else "Assistant"
        lines.append(f"{label}: {m.content}")
    return "\n".join(lines)


def build_retrieval_hint(history: List[Message]) -> str:
    """
    Build a short context hint for retrieval from the last 2 messages.

    We use a SHORT slice here (not the full history) because
    retrieval works better with focused queries. Sending the full
    history as the search query makes it too noisy and returns
    irrelevant chunks.

    Example:
        History: [..., "User: what is NPL?", "Assistant: NPL is..."]
        Hint: "what is NPL? NPL is..."

    This hint is combined with the current question so follow-up
    questions like "who won the first season?" retrieve NPL chunks
    instead of random cricket chunks.
    """
    if not history:
        return ""
    last_two = history[-2:]
    return " ".join([m.content for m in last_two])


# ── Health Check ─────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Simple health check endpoint.
    React frontend can ping this to verify backend is running.
    Visit http://localhost:8000/health to check server status.
    """
    return {"status": "running", "version": "1.0.0"}


# ── Main Q&A Endpoint ────────────────────────────────────────
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    The core endpoint. Receives a cricket question + conversation
    history, runs the full RAG pipeline, and streams the answer
    back token by token.

    Pipeline:
        1. Build history context
        2. Fetch missing Wikipedia knowledge dynamically
        3. Hybrid retrieval (vector + BM25)
        4. Rerank chunks by relevance
        5. Validate context is sufficient
        6. Stream answer token by token

    Returns a StreamingResponse with newline-delimited JSON events:
        {"type": "sources", "data": [...]}  — retrieved chunks
        {"type": "token",   "data": "word"} — each answer token
        {"type": "done"}                    — stream complete
    """
    llm = get_llm()

    def stream_response():
        try:
            print(f"\n{'='*50}")
            print(f"Question: {request.question}")
            print(f"History : {len(request.history)} messages")

            # ── Step 1: Build history strings ────────────────
            # Two different history formats for two different purposes:
            # - history_text: full context for the LLM answer prompt
            # - retrieval_hint: short context for better chunk retrieval
            history_text = build_history_text(request.history)
            retrieval_hint = build_retrieval_hint(request.history)

            # ── Step 2: Dynamic knowledge fetching ───────────
            # If user asks about something not in our DB,
            # this fetches the Wikipedia article automatically.
            # Uses history so "who won the first season?"
            # after NPL discussion fetches NPL, not IPL.
            # Fails gracefully — if Wikipedia is slow/down,
            # we still try to answer from existing knowledge.
            ensure_knowledge(request.question, retrieval_hint)

            # ── Step 3: Hybrid retrieval ──────────────────────
            # Searches ChromaDB using both:
            # - Vector search: semantic similarity (finds meaning)
            # - BM25 search: keyword matching (finds exact terms)
            # Returns up to 12 unique chunks combined from both
            docs = hybrid_retrieve(request.question, retrieval_hint)

            # ── Step 4: Reranking ─────────────────────────────
            # LLM reads all 12 chunks and picks the 8 most relevant
            # This fixes cases where embedding model returns wrong
            # chunks (e.g. Terry Alderman when asking about wickets)
            docs = rerank_docs(request.question, docs)

            # Format chunks into context string for the prompt
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send sources to frontend so user can inspect them
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # ── Step 5: Validation Node ───────────────────────
            # This is the key hallucination prevention step.
            # Before answering, LLM checks: "Do I actually have
            # enough information to answer this accurately?"
            #
            # Without this: LLM guesses from irrelevant chunks
            # With this: LLM honestly says "I don't know"
            #
            # Returns: {"is_sufficient": bool, "confidence": str}
            validation = validate_context(request.question, context)

            if not validation["is_sufficient"]:
                # Context is too weak — give honest response
                # instead of a confident wrong answer
                print(f"Validation FAILED: {validation['reason']}")
                yield json.dumps({
                    "type": "token",
                    "data": "I don't have enough verified information in my knowledge base to answer that accurately. Try asking about a specific player, tournament, or cricket rule — I might know more about that."
                }) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return

            print(f"Validation PASSED: {validation['confidence']}")

            # ── Step 6: Build prompt and stream answer ────────
            # Prompt structure:
            # [strict rules] + [conversation history] + [context] + [question]
            #
            # We pass history so the LLM resolves references like
            # "this league", "him", "they" correctly.
            #
            # We name it "Knowledge base" not "Wikipedia context"
            # so the LLM doesn't say "based on the Wikipedia context..."
            prompt_text = f"""You are a cricket expert assistant.

STRICT RULES:
1. Answer ONLY using facts from your knowledge base below
2. NEVER mention "context", "Wikipedia", "sources" or "provided information"
3. NEVER explain where your information comes from or does not come from
4. NEVER say things like "based on the context" or "the context shows"
5. NEVER invent statistics, names, dates, or facts not in the knowledge base
6. Use conversation history to understand references like "this league", "him", "they", "it"
7. If asked about something not in the knowledge base, say:
   "I don't have information about that in my knowledge base."

Previous conversation:
{history_text}

Knowledge base:
{context}

Current Question: {request.question}

Answer:"""

            # Stream the answer token by token
            # Groq returns AIMessageChunk objects — we extract .content
            # Each chunk is sent immediately so the frontend can
            # display words as they arrive (like ChatGPT)
            for chunk in llm.stream(prompt_text):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:  # skip empty chunks
                    yield json.dumps({"type": "token", "data": token}) + "\n"

            # Signal to frontend that streaming is complete
            yield json.dumps({"type": "done"}) + "\n"
            print(f"Response complete")

        except Exception as e:
            # Never let an error crash silently
            # Always send an error token so frontend shows something
            print(f"Error in stream_response: {e}")
            yield json.dumps({
                "type": "token",
                "data": f"Something went wrong on the server. Please try again."
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    # StreamingResponse sends generator output incrementally
    # React reads this with response.body.getReader()
    # media_type="text/plain" so browser treats it as a stream
    return StreamingResponse(stream_response(), media_type="text/plain")


# ── Summarize History Helper ─────────────────────────────────
def summarize_old_history(history: List[Message], llm) -> str:
    """
    For very long conversations, summarize older messages
    instead of dropping them completely.

    Strategy:
    - Short history (≤12 messages): use directly, no summary needed
    - Long history (>12 messages): summarize old + keep recent verbatim

    This means the agent remembers the full conversation even
    after many turns — early context isn't just dropped.

    Example output:
        [Earlier: User asked about Nepal Premier League. Agent explained
        it's a T20 league founded in 2024 with 8 teams.]

        User: who won the first season?
        Assistant: Janakpur Bolts won the first season.
        User: what about the second season?
    """
    if len(history) <= 12:
        return build_history_text(history)

    # Split into old and recent
    old_messages = history[:-6]
    recent_messages = history[-6:]

    old_text = "\n".join([
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in old_messages
    ])

    summary_prompt = f"""Summarize this cricket conversation in 2-3 sentences.
Focus on the cricket topics discussed and key facts mentioned:

{old_text}

Summary:"""

    try:
        raw = llm.invoke(summary_prompt)
        summary = raw.content if hasattr(raw, "content") else str(raw)
        summary = summary.strip()
    except Exception:
        summary = ""

    recent_text = build_history_text(recent_messages)

    if summary:
        return f"[Earlier in conversation: {summary}]\n\n{recent_text}"
    return recent_text