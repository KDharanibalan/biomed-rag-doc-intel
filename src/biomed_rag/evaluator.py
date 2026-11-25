"""Quantitative retrieval and generation evaluator + reproducible ablation runner."""

import json
from pathlib import Path
from typing import Dict, List

from .pipeline import BiomedRAGPipeline
from .schemas import AblationMetrics


def evaluate_pipeline(
    pipeline: BiomedRAGPipeline,
    eval_cases: List[Dict],
    experiment_name: str,
    top_k: int = 4,
) -> AblationMetrics:
    """Measure Recall@K, MRR, Answer Keyword Coverage, Citation Precision, and Latency."""
    hits = 0
    rr_sum = 0.0
    kw_cov_sum = 0.0
    cite_prec_sum = 0.0
    latencies: List[float] = []

    for case in eval_cases:
        q = case["query"]
        expected_doc = case["expected_doc_id"]
        expected_kws = case["expected_keywords"]

        retrieved = pipeline.store.search(q, top_k=top_k, mode=pipeline.retrieval_mode)
        retrieved_docs = [r.chunk.doc_id for r in retrieved]

        if expected_doc in retrieved_docs:
            hits += 1
            first_rank = retrieved_docs.index(expected_doc) + 1
            rr_sum += 1.0 / first_rank

        resp = pipeline.query(q, top_k=top_k)
        latencies.append(resp.latency_ms)

        ans_lower = resp.answer.lower()
        matched = sum(1 for kw in expected_kws if kw.lower() in ans_lower)
        kw_cov_sum += matched / max(1, len(expected_kws))

        if resp.citations:
            correct_cites = sum(1 for c in resp.citations if c.doc_id == expected_doc)
            cite_prec_sum += correct_cites / len(resp.citations)

    n = max(1, len(eval_cases))
    return AblationMetrics(
        experiment_name=experiment_name,
        chunk_strategy=pipeline.chunk_strategy,
        retrieval_mode=pipeline.retrieval_mode,
        top_k=top_k,
        recall_at_k=round(hits / n, 4),
        mrr=round(rr_sum / n, 4),
        keyword_coverage=round(kw_cov_sum / n, 4),
        citation_precision=round(cite_prec_sum / n, 4),
        avg_latency_ms=round(sum(latencies) / n, 2),
        num_queries=len(eval_cases),
        num_chunks=len(pipeline.store.chunks),
    )


def run_ablation_matrix(corpus_dir: Path, eval_path: Path, top_k: int = 3) -> List[AblationMetrics]:
    """Compare chunking strategies x retrieval modes systematically."""
    eval_cases = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    configs = [
        ("fixed_bm25", "fixed", "bm25"),
        ("fixed_dense", "fixed", "dense"),
        ("sliding_hybrid", "sliding_window", "hybrid"),
        ("section_bm25", "section_aware", "bm25"),
        ("section_dense", "section_aware", "dense"),
        ("section_hybrid", "section_aware", "hybrid"),
    ]
    results: List[AblationMetrics] = []
    for exp_name, chunk_strat, ret_mode in configs:
        pipe = BiomedRAGPipeline(
            chunk_strategy=chunk_strat,
            retrieval_mode=ret_mode,
            top_k=top_k,
        )
        pipe.ingest_directory(corpus_dir)
        results.append(evaluate_pipeline(pipe, eval_cases, experiment_name=exp_name, top_k=top_k))
    return results
