"""Pydantic v2 schemas for document chunks, RAG outputs, citations, and evaluations."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single indexed text segment with provenance metadata."""

    chunk_id: str
    doc_id: str
    source_file: str
    doc_type: str = Field(description="Inferred document type: SOP, SDS, COA, or STUDY")
    section_title: str
    text: str
    token_count: int


class RetrievalResult(BaseModel):
    """A retrieved chunk paired with its similarity/ranking score."""

    chunk: DocumentChunk
    score: float
    retrieval_mode: str


class Citation(BaseModel):
    """Inline provenance reference linking an answer back to source text."""

    doc_id: str
    chunk_id: str
    section_title: str
    excerpt: str


class RAGResponse(BaseModel):
    """Citation-grounded response produced by the RAG pipeline."""

    query: str
    answer: str
    citations: List[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool
    latency_ms: float
    prompt_version: str
    retrieval_mode: str


class StructuredExtraction(BaseModel):
    """Schema-constrained structured record extracted from regulated documents."""

    doc_id: str
    doc_type: str
    title: str
    key_fields: Dict[str, str]
    safety_or_qc_requirements: List[str]
    citations: List[Citation]
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class AblationMetrics(BaseModel):
    """Reproducible evaluation summary for a single chunking/retrieval experiment."""

    experiment_name: str
    chunk_strategy: str
    retrieval_mode: str
    top_k: int
    recall_at_k: float
    mrr: float
    keyword_coverage: float
    citation_precision: float
    avg_latency_ms: float
    num_queries: int
    num_chunks: int
    notes: Optional[str] = None
