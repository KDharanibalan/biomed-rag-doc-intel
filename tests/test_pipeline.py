"""Unit and integration tests for biomed-rag-doc-intel."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomed_rag.api import app
from biomed_rag.evaluator import run_ablation_matrix
from biomed_rag.pipeline import BiomedRAGPipeline


def test_section_aware_hybrid_retrieval_and_citation():
    pipe = BiomedRAGPipeline.from_config(ROOT / "configs" / "rag_config.yaml")
    n_chunks = pipe.ingest_directory(ROOT / "data" / "sample_corpus")
    assert n_chunks >= 12

    resp = pipe.query("What ultracentrifugation speed and duration are specified for EV pelleting?")
    assert resp.grounded is True
    assert "100,000 x g" in resp.answer
    assert any(c.doc_id == "SOP-BIO-EV-014" for c in resp.citations)


def test_refusal_on_unanswerable_query():
    pipe = BiomedRAGPipeline.from_config(ROOT / "configs" / "rag_config.yaml")
    pipe.ingest_directory(ROOT / "data" / "sample_corpus")

    resp = pipe.query("What is the orbital inclination of Jupiter's moon Europa?")
    assert resp.grounded is False
    assert "INSUFFICIENT_EVIDENCE" in resp.answer


def test_structured_coa_extraction():
    pipe = BiomedRAGPipeline.from_config(ROOT / "configs" / "rag_config.yaml")
    pipe.ingest_directory(ROOT / "data" / "sample_corpus")

    extraction = pipe.extract_structured("COA-BIO-CLP-2026-042")
    assert extraction.doc_type == "COA"
    assert extraction.key_fields["Batch / Lot Number"] == "CLP-2026-B042"
    assert any("97.4%" in val for val in extraction.key_fields.values())
    assert extraction.extraction_confidence >= 0.9


def test_ablation_matrix_meets_quality_floor():
    metrics = run_ablation_matrix(
        ROOT / "data" / "sample_corpus",
        ROOT / "data" / "eval_qa_dataset.json",
        top_k=3,
    )
    hybrid = next(m for m in metrics if m.experiment_name == "section_hybrid")
    assert hybrid.recall_at_k >= 0.85
    assert hybrid.mrr >= 0.80


def test_fastapi_endpoints():
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    q = client.post("/query", json={"query": "What GHS hazard code applies to Dexamethasone?"})
    assert q.status_code == 200
    body = q.json()
    assert body["grounded"] is True
    assert "H360" in body["answer"]

    e = client.get("/extract/SDS-CHEM-DEX-009")
    assert e.status_code == 200
    assert e.json()["doc_type"] == "SDS"
