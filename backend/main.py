import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import ask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer, sources = ask(request.question)
    return AnswerResponse(answer=answer, sources=sources)