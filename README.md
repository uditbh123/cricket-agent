# Cricket AI Agent

A full-stack AI-powered Q&A agent that answers questions about cricket using Wikipedia as its knowledge base. Built with a RAG (Retrieval-Augmented Generation) pipeline, a local LLM running on your own machine, and a React frontend no API keys, no cloud costs, completely free to run.

---

## What It Does

You type a cricket question "How does the Duckworth-Lewis method work?" or "Who holds the record for most Test centuries?" and the agent finds the most relevant information from a pre-indexed set of Wikipedia articles and generates a grounded, accurate answer using a locally running AI model.

It doesn't guess. It retrieves first, then answers. That's what makes it reliable.

---

## How It Works (The Architecture)

```
User asks a question
        ↓
React frontend sends POST request to FastAPI backend
        ↓
Question is converted into an embedding (a vector of numbers representing its meaning)
        ↓
ChromaDB searches for the most semantically similar chunks from Wikipedia articles
        ↓
Top 5 relevant chunks are passed to LLaMA 3.2 (running locally via Ollama) as context
        ↓
LLM reads the context and generates a grounded answer
        ↓
Answer + source chunks are returned to the React frontend
```

This pattern is called **RAG (Retrieval-Augmented Generation)**. It's the most widely used architecture for building production AI assistants that need to answer questions from a specific knowledge base without hallucinating.

---

## Tech Stack

### Backend
| Technology | What It Does |
|---|---|
| **Python** | Core language for all backend logic |
| **FastAPI** | REST API framework exposes `/ask` and `/health` endpoints |
| **LangChain** | Orchestrates the RAG pipeline (retrieval + prompt + LLM) |
| **Ollama + LLaMA 3.2** | Local LLM runs entirely on your machine, no API key needed |
| **ChromaDB** | Local vector database stores and searches Wikipedia embeddings |
| **HuggingFace Sentence Transformers** | Converts text into embeddings (`all-MiniLM-L6-v2` model) |
| **wikipedia-api** | Fetches Wikipedia article content programmatically |

### Frontend
| Technology | What It Does |
|---|---|
| **React + Vite** | Component-based UI with fast development build |
| **Axios** | HTTP client for communicating with the FastAPI backend |
| **Plain CSS** | Custom dark-mode chat interface |

---

## Project Structure

```
cricket-agent/
│
├── backend/
│   ├── ingest.py        # Fetches Wikipedia articles, creates embeddings, builds ChromaDB
│   ├── agent.py         # RAG pipeline — retriever + prompt + LLM chain
│   └── main.py          # FastAPI server — exposes the API endpoints
│
├── frontend/
│   └── src/
│       ├── App.jsx              # Root component, manages chat state
│       ├── App.css              # All styles
│       └── components/
│           ├── ChatWindow.jsx   # Renders message history + suggestion prompts
│           ├── ChatMessage.jsx  # Individual message bubble with source toggle
│           └── ChatInput.jsx    # Textarea + send button with keyboard shortcut
│
└── .gitignore
```

---

## Running It Locally

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- [Ollama](https://ollama.com) installed

### 1. Clone the repo

```bash
git clone https://github.com/uditbh123/cricket-agent.git
cd cricket-agent
```

### 2. Set up the Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
venv\Scripts\pip.exe install langchain langchain-community langchain-ollama langchain-huggingface langchain-chroma langchain-text-splitters chromadb sentence-transformers wikipedia fastapi uvicorn python-multipart
```

### 4. Pull the LLM

```bash
ollama pull llama3.2
```

This downloads LLaMA 3.2 (~2GB) to your machine. Only needed once.

### 5. Build the knowledge base

```bash
venv\Scripts\python.exe backend\ingest.py
```

This fetches ~13 Wikipedia articles about cricket, splits them into chunks, embeds them, and saves everything into a local ChromaDB database at `backend/cricket_db/`. Takes 2–5 minutes on first run.

### 6. Start the backend

```bash
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

API is now live at `http://localhost:8000`. Visit `http://localhost:8000/docs` to see the auto-generated API documentation and test it interactively.

### 7. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Reference

### `GET /health`
Returns server status.

```json
{ "status": "running" }
```

### `POST /ask`
Accepts a cricket question, returns an AI-generated answer with source chunks.

**Request:**
```json
{ "question": "How many Test centuries has Sachin Tendulkar scored?" }
```

**Response:**
```json
{
  "answer": "Sachin Tendulkar scored 51 centuries in Test cricket...",
  "sources": [
    "Tendulkar holds the record for the most centuries in international cricket...",
    "..."
  ]
}
```

---

## Knowledge Base

The agent currently knows about these Wikipedia topics:

- Cricket (overview)
- Test cricket
- One Day International (ODI)
- Twenty20
- Indian Premier League (IPL)
- Laws of Cricket
- Sachin Tendulkar
- Virat Kohli
- Brian Lara
- Shane Warne
- ICC Cricket World Cup
- The Ashes
- Duckworth–Lewis–Stern method

To add more topics, open `backend/ingest.py`, add article titles to the `topics` list, and re-run the script.

---

## What I Learned Building This

- **RAG architecture** — how to build a retrieval pipeline that grounds LLM responses in real documents instead of letting it hallucinate
- **Vector embeddings** — how text gets converted into numerical representations that capture semantic meaning, enabling similarity search
- **FastAPI** — building a REST API in Python, handling CORS, defining request/response schemas with Pydantic
- **React fundamentals** — component structure, `useState`, `useEffect`, `useRef`, controlled inputs, and async data fetching
- **Full-stack integration** — making a React frontend communicate with a Python backend via HTTP
- **LangChain LCEL** — the modern pipe-based chain syntax for composing LLM pipelines
- **Local AI** — running a full LLM on a personal laptop with zero cloud dependency using Ollama

---

## What's Next

- [ ] Add conversation memory so the agent remembers context within a session
- [ ] Expand the knowledge base with more players, tournaments, and historical matches
- [ ] Stream the LLM response token by token instead of waiting for the full answer
- [ ] Deploy backend to Railway or Render, frontend to Vercel

---

## Author

**Udit** — ICT & Robotics student.

[GitHub](https://github.com/uditbh123)