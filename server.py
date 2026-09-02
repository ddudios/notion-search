from fastapi import FastAPI
from pydantic import BaseModel

import agent

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: Question) -> dict:
    answer = agent.run(payload.question)
    return {"answer": answer}