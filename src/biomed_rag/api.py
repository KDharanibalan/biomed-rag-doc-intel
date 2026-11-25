"""FastAPI microservice exposing Biomedical Document Intelligence endpoints for Digital Workers."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import BiomedRAGPipeline
from .schemas import RAGResponse, StructuredExtraction

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "rag_config.yaml"
CORPUS_DIR = ROOT / "data" / "sample_corpus"

app = FastAPI(
    title="Biomed RAG Document Intelligence API",
    version="0.1.0",
    description="Domain-adaptive RAG and structured extraction service for regulated scientific documents.",
)

_pipeline: Optional[BiomedRAGPipeline] = None


def get_pipeline() -> BiomedRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BiomedRAGPipeline.from_config(CONFIG_PATH)
        _pipeline.ingest_directory(CORPUS_DIR)
    return _pipeline


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    retrieval_mode: Optional[str] = None


@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "ok",
        "indexed_chunks": len(pipe.store.chunks),
        "chunk_strategy": pipe.chunk_strategy,
        "retrieval_mode": pipe.retrieval_mode,
    }


@app.post("/query", response_model=RAGResponse)
def query_documents(req: QueryRequest):
    pipe = get_pipeline()
    return pipe.query(req.query, top_k=req.top_k, mode=req.retrieval_mode)


@app.get("/extract/{doc_id}", response_model=StructuredExtraction)
def extract_document(doc_id: str):
    pipe = get_pipeline()
    try:
        return pipe.extract_structured(doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
