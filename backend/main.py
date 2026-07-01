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
# In production, replace * with your actual frontend URL.
# Example: allow_origins=["https://cricket-agent.vercel.app"]
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
    from getting too long. Beyond that, context gets noisy and the
    LLM loses focus on the current question.

    Format:
        User: what is the Nepal Premier League?
        Assistant: The Nepal Premier League is a T20 league...
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

    Uses last 4 messages with proper User/Assistant labels so the LLM
    can resolve follow-up references correctly.

    Why 4 messages and not 2:
        Old version joined last 2 messages as a raw string with no labels.
        The LLM couldn't tell what was asked vs what was answered, so
        follow-ups like "who won the first season of it?" failed to resolve
        "it" back to "Nepal Premier League".

    Why truncate assistant responses to 300 chars:
        We only need enough to identify the topic being discussed.
        Sending the full assistant response makes the retrieval query
        too long and noisy, returning irrelevant chunks.

    Example — what the LLM sees as retrieval_hint:
        User: what is NPL?
        Assistant: The Nepal Premier League (NPL) is a T20 league...
        User: who won the first season of it?

    With this structure, "it" clearly resolves to "Nepal Premier League".
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
        1. Build history strings (full for LLM prompt, short for retrieval)
        2. Dynamically fetch missing Wikipedia knowledge
        3. Hybrid retrieval — vector search + BM25 keyword search
        4. Rerank chunks by relevance using LLM
        5. Validate context is sufficient to answer
        6. Stream answer token by token (filtering Qwen reasoning blocks)

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
            # Two different history formats for two different purposes:
            #
            # history_text — full formatted history for the LLM answer prompt.
            # Lets the LLM resolve references like "this league", "him", "they".
            #
            # retrieval_hint — short structured snippet for retrieval.
            # Used by ensure_knowledge() and hybrid_retrieve() to resolve
            # follow-up questions like "who won the first season of it?"
            # back to the correct topic (e.g. Nepal Premier League).
            history_text = build_history_text(request.history)
            retrieval_hint = build_retrieval_hint(request.history)

            # ── Step 2: Dynamic knowledge fetching ───────────
            # If the question mentions something not in our DB,
            # this fetches the Wikipedia article and stores it permanently.
            # Uses retrieval_hint so follow-up questions fetch the right topic.
            # Fault-tolerant — if Wikipedia is slow or down, we still
            # attempt to answer from existing knowledge.
            ensure_knowledge(request.question, retrieval_hint)

            # ── Step 3: Hybrid retrieval ──────────────────────
            # Combines two complementary search strategies:
            #
            # Vector search (ChromaDB): semantic similarity
            #   "little master" → finds Sachin Tendulkar chunks
            #   "fastest bowler" → finds Shoaib Akhtar chunks
            #
            # BM25: exact keyword matching
            #   "Muralitharan 800 wickets" → finds chunks with those exact words
            #
            # Returns up to 12 unique chunks combined from both.
            docs = hybrid_retrieve(request.question, retrieval_hint)

            # ── Step 4: Reranking ─────────────────────────────
            # LLM reads all 12 chunks and picks the 8 most relevant.
            # Fixes cases where the embedding model retrieves wrong chunks.
            # Example: asking "most Test wickets" without reranking might
            # return Terry Alderman chunks instead of Muralitharan chunks.
            # Adds ~0.5s latency but dramatically improves answer accuracy.
            docs = rerank_docs(request.question, docs)

            # Format chunks into a single context string.
            # format_docs also deduplicates — hybrid search often returns
            # the same chunk from both vector and BM25 searches.
            sources = [doc.page_content for doc in docs]
            context = format_docs(docs)

            # Send sources to frontend so user can inspect retrieved chunks.
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # ── Step 5: Validation Node ───────────────────────
            # Key hallucination prevention step.
            # LLM checks: "Is this context actually sufficient to answer?"
            #
            # HIGH / MEDIUM → proceed to generate answer
            # LOW → return honest "I don't know" response
            #
            # Without this: LLM confidently hallucinates from irrelevant chunks.
            # With this: LLM honestly admits when context is insufficient.
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

            # ── Step 6: Build prompt ──────────────────────────
            # Prompt structure:
            #   [strict rules] + [conversation history] + [context] + [question]
            #
            # history_text lets the LLM resolve pronouns and vague references.
            # We call it "Knowledge base" not "Wikipedia context" so the LLM
            # doesn't leak phrases like "based on Wikipedia..." into answers.
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

            # ── Step 7: Stream answer (with Qwen think-block filter) ──
            # Qwen 3.6 is a reasoning model. Before answering, it thinks
            # out loud inside <think>...</think> tags. This internal reasoning
            # is useful for accuracy but must never be shown to users.
            #
            # Strategy:
            # - Buffer all tokens until </think> is detected
            # - Once </think> appears, stream only what comes after it
            # - If no <think> block at all (model skipped reasoning),
            #   flush the buffer after 50 chars and stream normally
            #
            # This means the first few tokens are slightly delayed
            # (buffered during thinking) but the final answer streams
            # normally token by token.

            buffer = ""         # accumulates tokens during think block
            in_think_block = False
            think_done = False  # True once we've passed </think>

            for chunk in llm.stream(prompt_text):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not token:
                    continue

                if not think_done:
                    buffer += token

                    # Detect opening of think block
                    if "<think>" in buffer:
                        in_think_block = True

                    # Detect closing of think block
                    if "</think>" in buffer:
                        in_think_block = False
                        think_done = True
                        # Split on </think> and take everything after it
                        # lstrip removes leading newlines Qwen adds after thinking
                        after_think = buffer.split("</think>", 1)[-1].lstrip("\n")
                        if after_think:
                            yield json.dumps({"type": "token", "data": after_think}) + "\n"
                        continue

                    # Model skipped reasoning entirely — no <think> tag found
                    # after buffering 50 chars. Flush buffer and stream normally.
                    if not in_think_block and len(buffer) > 50 and "<think>" not in buffer:
                        think_done = True
                        yield json.dumps({"type": "token", "data": buffer}) + "\n"

                    continue

                # Think block is done — stream every subsequent token immediately
                yield json.dumps({"type": "token", "data": token}) + "\n"

            # Signal to frontend that streaming is complete.
            yield json.dumps({"type": "done"}) + "\n"
            print("Response complete")

        except Exception as e:
            # Never crash silently — always send an error to the frontend
            # so the user sees something instead of a frozen spinner.
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

    NOT YET WIRED INTO THE MAIN FLOW.
    To enable: in stream_response(), replace:
        history_text = build_history_text(request.history)
    with:
        history_text = summarize_old_history(request.history, llm)

    Example output this produces:
        [Earlier in conversation: User asked about Nepal Premier League.
        Agent explained it's a T20 franchise league with 8 teams.]

        User: who won the first season?
        Assistant: Janakpur Bolts won the first season.
        User: what about the second season?
    """
    if len(history) <= 12:
        return build_history_text(history)

    # Split into old (to summarize) and recent (to keep verbatim)
    old_messages = history[:-6]
    recent_messages = history[-6:]

    old_text = "\n".join([
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in old_messages
    ])

    summary_prompt = f"""Summarize this cricket conversation in 2-3 sentences.
Focus on the cricket topics discussed and key facts mentioned.
Be specific — include names, tournaments, and statistics if mentioned.

{old_text}

Summary:"""

    try:
        raw = llm.invoke(summary_prompt)
        summary = raw.content if hasattr(raw, "content") else str(raw)
        # Strip any <think> blocks from Qwen reasoning model
        if "</think>" in summary:
            summary = summary.split("</think>", 1)[-1]
        summary = summary.strip()
    except Exception:
        summary = ""

    recent_text = build_history_text(recent_messages)

    if summary:
        return f"[Earlier in conversation: {summary}]\n\n{recent_text}"
    return recent_text