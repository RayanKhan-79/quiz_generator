from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from backend.schemas import GenerateRequest, GenerateResponse, VerifyRequest, VerifyResponse
from src.inference import QuizEngine

app = FastAPI(title="RACE TF-IDF Quiz Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = QuizEngine()


@app.get("/health")
def health():
    return engine.health()


@app.get("/sample")
def sample(debug: bool = Query(False)):
    try:
        return engine.sample(debug=debug)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest):
    try:
        return engine.generate(
            article=payload.article,
            question=payload.question,
            options=payload.options.as_dict() if payload.options else None,
            question_count=payload.question_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest):
    try:
        return engine.verify(payload.article, payload.question, payload.options.as_dict(), payload.selected_option)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics():
    return engine.metrics()


@app.get("/logs/export", response_class=PlainTextResponse)
def export_logs():
    return PlainTextResponse(engine.store.csv(), media_type="text/csv")
