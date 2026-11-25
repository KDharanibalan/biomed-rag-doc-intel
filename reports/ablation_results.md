# Reproducible RAG Ablation Benchmark (top_k=3)

| Experiment | Chunking | Retrieval | Chunks | Recall@3 | MRR | Keyword Coverage | Citation Precision | Avg Latency (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_bm25` | fixed | bm25 | 9 | 1.000 | 1.000 | 0.667 | 0.958 | 1.06 |
| `fixed_dense` | fixed | dense | 9 | 1.000 | 1.000 | 0.667 | 1.000 | 2.21 |
| `sliding_hybrid` | sliding_window | hybrid | 10 | 1.000 | 1.000 | 0.667 | 1.000 | 2.34 |
| `section_bm25` | section_aware | bm25 | 24 | 1.000 | 1.000 | 0.861 | 0.931 | 0.53 |
| `section_dense` | section_aware | dense | 24 | 1.000 | 1.000 | 0.861 | 0.931 | 1.65 |
| `section_hybrid` | section_aware | hybrid | 24 | 1.000 | 1.000 | 0.861 | 0.931 | 3.86 |
