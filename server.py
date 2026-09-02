from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import time
from collections import defaultdict

import agent
import os
from dotenv import load_dotenv

load_dotenv()
MY_SECRET_API_KEY = os.environ["MY_SECRET_API_KEY"]

app = FastAPI()
request_log: dict = defaultdict(list)
RATE_LIMIT = 5
WINDOW_SECONDS = 60

def check_rate_limit(key: str) -> None:
    now = time.time()
    request_log[key] = [t for t in request_log[key] if now - t < WINDOW_SECONDS]
    if len(request_log[key]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="너무 많은 요청입니다. 잠시 후 다시 시도해주세요.")
    request_log[key].append(now)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: Question, x_api_key: str = Header(...)) -> dict:
    if x_api_key != MY_SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="인증 실패")
    check_rate_limit(x_api_key)
    answer = agent.run(payload.question)
    return {"answer": answer}