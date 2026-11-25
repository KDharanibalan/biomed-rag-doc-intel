#!/usr/bin/env python3
"""Run reproducible chunking x retrieval ablations and save JSON + Markdown reports."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomed_rag.evaluator import run_ablation_matrix


def main() -> None:
    corpus_dir = ROOT / "data" / "sample_corpus"
    eval_path = ROOT / "data" / "eval_qa_dataset.json"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    metrics = run_ablation_matrix(corpus_dir, eval_path, top_k=3)

    json_out = reports_dir / "ablation_results.json"
    json_out.write_text(
        json.dumps([m.model_dump() for m in metrics], indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Reproducible RAG Ablation Benchmark (top_k=3)",
        "",
        "| Experiment | Chunking | Retrieval | Chunks | Recall@3 | MRR | Keyword Coverage | Citation Precision | Avg Latency (ms) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics:
        lines.append(
            f"| `{m.experiment_name}` | {m.chunk_strategy} | {m.retrieval_mode} | {m.num_chunks} "
            f"| {m.recall_at_k:.3f} | {m.mrr:.3f} | {m.keyword_coverage:.3f} "
            f"| {m.citation_precision:.3f} | {m.avg_latency_ms:.2f} |"
        )
    md_text = "\n".join(lines) + "\n"
    (reports_dir / "ablation_results.md").write_text(md_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
