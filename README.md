# 🏏 Cricket AI Agent

A full-stack AI-powered cricket Q&A system built with RAG (Retrieval-Augmented Generation). Ask anything about cricket — rules, players, tournaments, records, history — and get grounded answers sourced directly from Wikipedia. Runs entirely free on your local machine with no API keys or cloud costs.

---

## What Makes This Different

Most AI chatbots answer from training data and hallucinate when they don't know something. This agent retrieves real Wikipedia content first, then generates answers from that content — so every answer is traceable to a source.

The knowledge base is self-maintaining. It crawls Wikipedia's own cricket category system to discover articles automatically, builds the database incrementally (never wiping existing data), and grows permanently with every run. You never manually list topics.

---

## How It Works

```
User asks a question
        ↓
LLM extracts Wikipedia search topics from the question
(handles nicknames, slang, vague phrasing — anything)
        ↓
knowledge_manager checks if those topics are already indexed
        ↓
If new → fetches from Wikipedia API → chunks → embeds → stores in ChromaDB
If already indexed → skips fetch entirely
        ↓
Hybrid retrieval: Vector search (semantic) + BM25 (keyword) combined
        ↓
Top 12 deduplicated chunks sent to LLaMA 3.2 as context
        ↓
LLM generates a grounded answer strictly from the retrieved context
        ↓
Answer streams token by token to the React frontend
```

---

## Architecture

```
cricket-agent/
│
├── backend/
│   ├── crawler.py           # Crawls Wikipedia category system to discover articles
│   ├── ingest.py            # Incremental DB builder — never wipes existing data
│   ├── knowledge_manager.py # Fetches Wikipedia articles, manages fetched_topics.json
│   ├── agent.py             # Hybrid RAG pipeline (vector + BM25 retrieval)
│   ├── main.py              # FastAPI server with streaming responses
│   ├── cricket_db/          # ChromaDB vector database (gitignored)
│   ├── fetched_topics.json  # Tracks indexed articles (gitignored)
│   └── discovered_articles.json  # Crawler output — all discovered article titles
│
└── frontend/
    └── src/
        ├── App.jsx                  # Root component, streaming chat state
        ├── App.css                  # Dark theme styles
        └── components/
            ├── ChatWindow.jsx       # Message history + suggestion prompts
            ├── ChatMessage.jsx      # Message bubble with copy + sources toggle
            └── ChatInput.jsx        # Textarea with character counter
```

---

## Tech Stack

### Backend
| Technology | Role |
|---|---|
| Python | Core language |
| FastAPI | REST API — `/ask` (streaming) and `/health` endpoints |
| LangChain | RAG pipeline orchestration |
| Ollama + LLaMA 3.2 1b | Local LLM — runs on your machine, zero cost |
| ChromaDB | Local vector database for semantic search |
| HuggingFace Sentence Transformers | Text embeddings (`all-MiniLM-L6-v2`) |
| BM25 Retriever | Keyword-based retrieval to complement vector search |
| Wikipedia REST API | Article fetching via `requests` (no library dependency) |

### Frontend
| Technology | Role |
|---|---|
| React + Vite | Component-based UI |
| Fetch API | Streaming HTTP responses from FastAPI |
| Plain CSS | Custom dark theme chat interface |

---

## Key Engineering Decisions

**Hybrid Search (Vector + BM25)**
Pure vector search misses exact keyword matches. Pure keyword search misses semantic meaning. Combining both retrieves more relevant chunks — especially for cricket-specific terminology and player names.

**Streaming Responses**
The FastAPI backend streams LLM output token by token using `StreamingResponse`. The React frontend reads the stream incrementally, so words appear as they're generated instead of waiting for the full answer. This is the same pattern used by ChatGPT and Claude.

**Incremental Knowledge Base**
The database never gets deleted. `fetched_topics.json` tracks every indexed article. Running `ingest.py` again skips already-indexed articles and only fetches new ones. Failed fetches are automatically retried on the next run.

**Category-Based Discovery**
Instead of manually listing article titles, `crawler.py` crawls Wikipedia's cricket category system (`Category:Cricket`, `Category:Indian cricketers`, etc.) and discovers articles automatically. Adding a new category like `Category:Cricket governing bodies` pulls all articles in it with zero manual work.

**Dynamic Knowledge Gaps**
When a user asks about a topic not in the database, the agent extracts Wikipedia search topics from the question using the LLM, fetches those articles, and adds them to the database permanently. The knowledge base grows with usage.

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed

### 1. Clone and set up Python environment

```bash
git clone https://github.com/uditbh123/cricket-agent.git
cd cricket-agent

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-chroma langchain-text-splitters chromadb sentence-transformers fastapi uvicorn python-multipart requests
```

### 3. Pull the local LLM

```bash
ollama pull llama3.2:1b
```

Downloads LLaMA 3.2 1b (~700MB) to your machine. Only needed once.

### 4. Discover cricket articles from Wikipedia

```bash
python backend/crawler.py
```

Crawls Wikipedia's cricket category system and saves discovered article titles to `backend/discovered_articles.json`. No rate limits hit — only fetches category listings, not full articles.

### 5. Build the knowledge base

```bash
python backend/ingest.py
```

Fetches full Wikipedia articles for discovered topics, chunks them, embeds them, and stores in ChromaDB. Safe to run multiple times — already-indexed articles are skipped automatically.

### 6. Start the backend

```bash
# Windows
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

# Mac/Linux
uvicorn backend.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Expanding the Knowledge Base

### Add a new Wikipedia category

Open `backend/crawler.py` and add to `CRICKET_CATEGORIES`:

```python
CRICKET_CATEGORIES = [
    "Cricket",
    "Indian cricketers",
    "Cricket governing bodies",   # ← add this
    "Cricket stadiums",           # ← and this
]
```

Then run:
```bash
python backend/crawler.py
python backend/ingest.py
```

The crawler discovers all articles in the new categories. Ingest fetches only the ones not already in the database. Your existing data is never touched.

### The database never needs to be deleted

Running `ingest.py` is always safe. It checks `fetched_topics.json` before every fetch and skips anything already indexed.

---

## API Reference

### `GET /health`
```json
{ "status": "running" }
```

### `POST /ask`
Streams the response as newline-delimited JSON.

**Request:**
```json
{ "question": "Who took the most Test wickets ever?" }
```

**Stream events:**
```json
{ "type": "sources", "data": ["chunk text...", "..."] }
{ "type": "token", "data": "Muttiah" }
{ "type": "token", "data": " Muralitharan" }
{ "type": "done" }
```

---

## What I Learned Building This

**RAG architecture** — how to build a retrieval pipeline that grounds LLM responses in real documents and prevents hallucination by forcing the model to answer only from retrieved context.

**Hybrid search** — why combining vector similarity search with BM25 keyword search produces significantly better retrieval than either alone, especially for domain-specific terminology.

**Streaming with FastAPI + React** — how to stream LLM output from a Python backend to a React frontend using `StreamingResponse` and the browser's `ReadableStream` API.

**Incremental database design** — how to build a knowledge base that grows safely over time without ever needing to be wiped and rebuilt from scratch.

**Wikipedia as a data source** — how to use Wikipedia's REST API and category system programmatically to build a self-maintaining, domain-specific knowledge base without hardcoding topics.

**LangChain LCEL** — the modern pipe-based chain syntax for composing LLM pipelines compared to the older chain classes.

**Full-stack integration** — connecting a React frontend to a Python backend, handling CORS, async streaming, and state management for a real-time chat interface.

---

## Roadmap

- [ ] Deploy backend to Render, frontend to Vercel
- [ ] Add conversation memory so the agent remembers context within a session
- [ ] Scheduled crawler runs to keep the knowledge base fresh automatically
- [ ] Switch to a larger LLM when deploying to cloud for more accurate answers

---

## Author

**Udit** — ICT & Robotics student building real AI engineering projects from scratch.

[GitHub](https://github.com/uditbh123)