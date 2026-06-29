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
# to FastAPI (port 8000) — same-origin policy blocks cross-port requests.
# allow_origins=["*"] is fine for development.
# In production, replace * with your actual frontend URL (e.g. https://cricket-agent.vercel.app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Models ──────────────────────────────────────────────
# Pydantic validates incoming JSON automatically.
# Wrong type = 422 error returned to client.
# These models also generate the schema visible at /docs.

class Message(BaseModel):
    """A single message in the conversation history."""
    role: str       # "user" or "assistant"
    content: str    # the actual message text

class QuestionRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str                   # current question from user
    history: List[Message] = []     # full conversation so far


# ── Helper: Full History for LLM Prompt ─────────────────────
def build_history_text(history: List[Message], max_messages: int = 12) -> str:
    """
    Convert conversation history into a formatted string for the LLM answer prompt.

    Limited to last 12 messages (6 full turns) to keep the prompt
    from getting too long. Beyond that, context gets noisy.

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


# ── Helper: Short Context for Retrieval ─────────────────────
def build_retrieval_hint(history: List[Message]) -> str:
    """
    Build a short structured context hint for retrieval and topic extraction.

    Uses last 4 messages (not 2) with proper User/Assistant labels so the
    LLM can resolve follow-up references correctly.

    Example — what the LLM sees:
        User: what is NPL?
        Assistant: The Nepal Premier League (NPL) is a T20 league...
        User: who won the first season of it?

    With this structure, "it" clearly resolves to "Nepal Premier League".
    Without labels (old version), the LLM couldn't tell what was asked
    vs what was answered, so follow-ups failed.

    Assistant responses are truncated to 300 chars — we only need enough
    to identify the topic being discussed, not the full answer.
    """
    if not history:
        return ""
    last_four = history[-4:]
    lines = []
    for m in last_four:
        label = "User" if m.role == "user" else "Assistant"
        content = m.content[:300] if m.role == "assistant" else m.content
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


# ── Health Check ─────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Simple health check endpoint.
    Visit http://localhost:8000/health to verify the server is running.
    """
    return {"status": "running", "version": "1.0.0"}


# ── Main Q&A Endpoint ────────────────────────────────────────
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Core endpoint. Receives a cricket question + conversation history,
    runs the full RAG pipeline, and streams the answer token by token.

    Pipeline:
        1. Build history strings (full for LLM, short for retrieval)
        2. Dynamically fetch missing Wikipedia knowledge
        3. Hybrid retrieval — vector search + BM25 keyword search
        4. Rerank chunks by relevance using LLM
        5. Validate context is sufficient to answer
        6. Stream answer token by token

    Stream events (newline-delimited JSON):
        {"type": "sources", "data": [...]}   — retrieved chunks
        {"type": "token",   "data": "word"}  — each answer token
        {"type": "done"}                     — stream complete
    """
    llm = get_llm()

    def stream_response():
        try:
            print(f"\n{'='*50}")
            print(f"Question: {request.question}")
            print(f"History : {len(request.history)} messages")

            # ── Step 1: Build history strings ────────────────
            # history_text — full formatted history for the LLM answer prompt.
            # Lets the LLM understand references like "this league", "him", "they".
            #
            # retrieval_hint — short structured snippet for retrieval.
            # Used by ensure_knowledge and hybrid_retrieve to resolve
            # follow-up questions like "who won the first season of it?"
            # back to the correct topic (e.g. Nepal Premier League).
            history_text = build_history_text(request.history)
            retrieval_hint = build_retrieval_hint(request.history)

            # ── Step 2: Dynamic knowledge fetching ───────────
            # If the question is about something not in our DB,
            # this fetches the Wikipedia article and adds it permanently.
            # Uses retrieval_hint so follow-up questions fetch the right topic.
            # Fault-tolerant — if Wikipedia is slow/down, we still answer
            # from existing knowledge.
            ensure_knowledge(request.question, retrieval_hint)

            # ── Step 3: Hybrid retrieval ──────────────────────
            # Combines two complementary search strategies:
            # - Vector search (ChromaDB): semantic similarity
            #   "little master" → finds Sachin Tendulkar chunks
            # - BM25: exact keyword matching
            #   "Muralitharan 800 wickets" → finds exact chunks
            # Returns up to 12 unique chunks combined from both.
            docs = hybrid_retrieve(request.question, retrieval_hint)

            # ── Step 4: Reranking ─────────────────────────────
            # LLM reads all 12 chunks and picks the 8 most relevant.
            # Fixes cases where the embedding model retrieves wrong chunks
            # (e.g. Terry Alderman chunks when asking about most Test wickets).
            # Adds ~0.5s but dramatically improves answer accuracy.
            docs = rerank_docs(request.question, docs)

            # Format chunks into a single context string for the prompt.
            # format_docs also deduplicates — hybrid search often returns
            # the same chunk from both vector and BM25.
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send sources to frontend so user can inspect retrieved chunks.
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # ── Step 5: Validation Node ───────────────────────
            # Key hallucination prevention step.
            # LLM checks: "Is this context actually sufficient to answer?"
            #
            # HIGH / MEDIUM → proceed to answer
            # LOW → return honest "I don't know" instead of a wrong answer
            #
            # Without this: LLM confidently hallucinates from irrelevant chunks.
            # With this: LLM honestly admits when it doesn't have enough info.
            validation = validate_context(request.question, context)

            if not validation["is_sufficient"]:
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
            #   [strict rules] + [conversation history] + [context] + [question]
            #
            # history_text lets the LLM resolve pronouns and references.
            # We call it "Knowledge base" not "Wikipedia context" so the LLM
            # doesn't leak source mentions into its answers.
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

            # Stream answer token by token.
            # Groq returns AIMessageChunk objects — we extract .content from each.
            # Each token is sent immediately so the frontend displays words
            # as they arrive, like ChatGPT.
            for chunk in llm.stream(prompt_text):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield json.dumps({"type": "token", "data": token}) + "\n"

            # Signal to frontend that streaming is complete.
            yield json.dumps({"type": "done"}) + "\n"
            print("Response complete")

        except Exception as e:
            # Never crash silently — always send something to the frontend.
            print(f"Error in stream_response: {e}")
            yield json.dumps({
                "type": "token",
                "data": "Something went wrong on the server. Please try again."
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    # StreamingResponse sends generator output incrementally.
    # React reads this with response.body.getReader().
    # media_type="text/plain" tells the browser to treat it as a stream.
    return StreamingResponse(stream_response(), media_type="text/plain")


# ── Long Conversation Summarization ─────────────────────────
def summarize_old_history(history: List[Message], llm) -> str:
    """
    For very long conversations, summarize older messages instead of
    dropping them. Keeps full context without blowing up the prompt size.

    Strategy:
    - Short history (≤12 messages): use directly, no summary needed
    - Long history (>12 messages): summarize old + keep last 6 verbatim

    This function exists but is not yet wired into the main request flow.
    To enable: replace build_history_text(request.history) in stream_response
    with summarize_old_history(request.history, llm).

    Example output:
        [Earlier in conversation: User asked about Nepal Premier League.
        Agent explained it's a T20 league with 8 franchise teams.]

        User: who won the first season?
        Assistant: Janakpur Bolts won the first season.
        User: what about the second season?
    """
    if len(history) <= 12:
        return build_history_text(history)

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