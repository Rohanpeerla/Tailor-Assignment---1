from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_query

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer, image = run_query(query.question)
    return {
        "answer": answer,
        "image": image
    }


@app.get("/health")
def health():
    return {"status": "ok"}
