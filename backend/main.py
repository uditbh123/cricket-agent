from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import ask

# Create the FastAPI app
app = FastAPI()

# Allow React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], #Vite's default port 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what a request looks like 
class QuestionRequest(BaseModel):
    question: str

# Define what a response looks like 
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

# Health check endpoin 
@app.get("/health")
def health():
    return {"status": "running"}

# Main endpoint - receives a question, returns answer 
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer, sources = ask(request.question)
    return AnswerResponse(answer=answer, sources=sources)