"""End-to-end Biomedical Document Intelligence & RAG Pipeline."""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .chunking import chunk_document, load_corpus_directory
from .generator import GroundedExtractiveBackend, HuggingFaceLocalBackend, OpenAICompatibleBackend
from .schemas import Citation, DocumentChunk, RAGResponse, StructuredExtraction
from .vector_store import HybridVectorStore


class BiomedRAGPipeline:
    """Modular RAG + schema-constrained extraction pipeline for regulated scientific documents."""

    def __init__(
        self,
        chunk_strategy: str = "section_aware",
        chunk_size_words: int = 140,
        overlap_words: int = 30,
        retrieval_mode: str = "hybrid",
        top_k: int = 4,
        embedding_backend: str = "tfidf_svd",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        generation_backend: str = "grounded_extractive",
        llm_model: str = "gpt-4o-mini",
        prompt_version: str = "v2_strict_grounding",
        seed: int = 42,
    ):
        self.chunk_strategy = chunk_strategy
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.prompt_version = prompt_version

        self.store = HybridVectorStore(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            seed=seed,
        )
        if generation_backend == "grounded_extractive":
            self.generator = GroundedExtractiveBackend()
        elif generation_backend == "openai_compatible":
            self.generator = OpenAICompatibleBackend(model=llm_model)
        elif generation_backend == "huggingface_local":
            self.generator = HuggingFaceLocalBackend(model_name=llm_model)
        else:
            raise ValueError(f"Unsupported generation_backend: {generation_backend}")

        self._raw_docs: Dict[str, str] = {}

    @classmethod
    def from_config(cls, config_path: Path) -> "BiomedRAGPipeline":
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return cls(
            chunk_strategy=cfg["chunking"]["strategy"],
            chunk_size_words=cfg["chunking"]["chunk_size_words"],
            overlap_words=cfg["chunking"]["overlap_words"],
            retrieval_mode=cfg["retrieval"]["mode"],
            top_k=cfg["retrieval"]["top_k"],
            embedding_backend=cfg["retrieval"].get("embedding_backend", "tfidf_svd"),
            embedding_model=cfg["retrieval"].get("embedding_model", "BAAI/bge-small-en-v1.5"),
            generation_backend=cfg["generation"]["backend"],
            llm_model=cfg["generation"].get("model", "gpt-4o-mini"),
            prompt_version=cfg["generation"].get("prompt_version", "v2_strict_grounding"),
            seed=cfg["project"].get("seed", 42),
        )

    def ingest_directory(self, corpus_dir: Path) -> int:
        chunks = load_corpus_directory(
            corpus_dir,
            strategy=self.chunk_strategy,
            chunk_size_words=self.chunk_size_words,
            overlap_words=self.overlap_words,
        )
        for path in sorted(Path(corpus_dir).glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            doc_chunks = chunk_document(raw, path.name, strategy=self.chunk_strategy)
            if doc_chunks:
                self._raw_docs[doc_chunks[0].doc_id] = raw
        self.store.add_chunks(chunks)
        return len(chunks)

    def query(self, question: str, top_k: Optional[int] = None, mode: Optional[str] = None) -> RAGResponse:
        t0 = time.perf_counter()
        active_mode = mode or self.retrieval_mode
        results = self.store.search(question, top_k=top_k or self.top_k, mode=active_mode)
        answer, citations, confidence, grounded = self.generator.generate(
            question, results, prompt_version=self.prompt_version
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=grounded,
            latency_ms=latency_ms,
            prompt_version=self.prompt_version,
            retrieval_mode=active_mode,
        )

    def extract_structured(self, doc_id: str) -> StructuredExtraction:
        """Extract schema-constrained metadata, key parameters, and compliance rules for a document."""
        doc_chunks: List[DocumentChunk] = [c for c in self.store.chunks if c.doc_id == doc_id]
        if not doc_chunks:
            raise KeyError(f"Document ID '{doc_id}' not found in indexed store.")

        raw = self._raw_docs.get(doc_id, "\n".join(c.text for c in doc_chunks))
        title_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else doc_id

        key_fields: Dict[str, str] = {}
        for k, v in re.findall(r"\*\*([A-Za-z0-9 /\-\(\)]+):\*\*\s*(.+)", raw):
            key_fields[k.strip()] = v.strip()

        # Also extract markdown analytical table rows (common in COAs & specifications)
        for line in raw.splitlines():
            if line.strip().startswith("|") and "PASS" in line.upper():
                cols = [col.strip() for col in line.strip("|").split("|")]
                if len(cols) >= 4:
                    key_fields[f"Test: {cols[0]}"] = f"{cols[3]} (Spec: {cols[2]}, Status: {cols[-1]})"

        safety_or_qc: List[str] = []
        for chunk in doc_chunks:
            if any(
                kw in chunk.section_title.lower()
                for kw in ("safety", "hazard", "quality", "ppe", "storage", "disposal", "audit")
            ):
                for line in chunk.text.splitlines():
                    cleaned = line.strip("- *•").strip()
                    if cleaned and not cleaned.startswith("Document:") and len(cleaned) > 20:
                        safety_or_qc.append(cleaned)

        citations = [
            Citation(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                section_title=c.section_title,
                excerpt=c.text[:180],
            )
            for c in doc_chunks[:3]
        ]

        return StructuredExtraction(
            doc_id=doc_id,
            doc_type=doc_chunks[0].doc_type,
            title=title,
            key_fields=key_fields,
            safety_or_qc_requirements=safety_or_qc[:8],
            citations=citations,
            extraction_confidence=0.95 if key_fields else 0.6,
        )
