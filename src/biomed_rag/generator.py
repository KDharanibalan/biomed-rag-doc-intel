"""Pluggable generation backends: OpenAI-compatible (OpenAI/vLLM/Ollama), HuggingFace, and Grounded Extractive."""

import os
import re
from typing import List, Tuple

import httpx

from .prompts import build_rag_messages
from .schemas import Citation, RetrievalResult


def _split_evidence_units(text: str) -> List[str]:
    """Split chunk text into crisp sentences or table rows for fine-grained citation grounding."""
    units: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Document:") or set(line) <= {"|", "-", ":", " "}:
            continue
        if line.startswith("|") and line.endswith("|"):
            units.append(line)
        else:
            for sent in re.split(r"(?<=[.!?])\s+", line):
                s = sent.strip("- *•").strip()
                if len(s) >= 12:
                    units.append(s)
    return units


def _token_overlap(query: str, candidate: str) -> float:
    q_toks = set(re.findall(r"[a-z0-9][a-z0-9\-\./%]*", query.lower()))
    c_toks = set(re.findall(r"[a-z0-9][a-z0-9\-\./%]*", candidate.lower()))
    stopwords = {
        "what", "which", "when", "where", "how", "is", "are", "the", "for", "and",
        "of", "in", "to", "a", "an", "be", "or", "by", "did", "much", "many",
    }
    q_toks -= stopwords
    if not q_toks:
        return 0.0
    return len(q_toks & c_toks) / len(q_toks)


class GroundedExtractiveBackend:
    """Deterministic, zero-hallucination evidence synthesizer for reproducible offline/CI runs."""

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt_version: str = "v2_strict_grounding",
        min_overlap: float = 0.15,
    ) -> Tuple[str, List[Citation], float, bool]:
        scored_units = []
        for rank, res in enumerate(results):
            rank_bonus = 1.0 / (1.0 + 0.45 * rank)
            chunk_ov = _token_overlap(query, res.chunk.text)
            header_ov = _token_overlap(query, f"{res.chunk.section_title} {res.chunk.doc_id}")
            for unit in _split_evidence_units(res.chunk.text):
                unit_ov = _token_overlap(query, unit)
                combined_ov = max(unit_ov, 0.45 * unit_ov + 0.55 * max(chunk_ov, header_ov))
                if unit_ov > 0 or (rank == 0 and chunk_ov >= 0.35):
                    score = (0.65 * unit_ov + 0.35 * chunk_ov) * rank_bonus
                    scored_units.append((score, combined_ov, unit, res))

        scored_units.sort(key=lambda x: x[0], reverse=True)
        selected = [item for item in scored_units if item[1] >= min_overlap][:3]

        if not selected:
            return (
                "INSUFFICIENT_EVIDENCE: The indexed documents do not specify this information.",
                [],
                0.0,
                False,
            )

        answer_parts: List[str] = []
        citations: List[Citation] = []
        seen_chunks = set()
        for _, _, unit, res in selected:
            c = res.chunk
            answer_parts.append(f"{unit} [{c.doc_id} | {c.chunk_id}]")
            if c.chunk_id not in seen_chunks:
                seen_chunks.add(c.chunk_id)
                citations.append(
                    Citation(
                        doc_id=c.doc_id,
                        chunk_id=c.chunk_id,
                        section_title=c.section_title,
                        excerpt=unit[:220],
                    )
                )

        top_overlap = selected[0][1]
        confidence = round(min(0.99, 0.45 + 0.54 * top_overlap), 3)
        return " ".join(answer_parts), citations, confidence, True


class OpenAICompatibleBackend:
    """Connects to OpenAI, Azure OpenAI, vLLM, or Ollama chat-completion endpoints."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt_version: str = "v2_strict_grounding",
    ) -> Tuple[str, List[Citation], float, bool]:
        if not self.api_key and "openai.com" in self.base_url:
            raise EnvironmentError("OPENAI_API_KEY must be set for OpenAICompatibleBackend.")
        messages = build_rag_messages(query, results, prompt_version=prompt_version)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()

        citations = [
            Citation(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                section_title=r.chunk.section_title,
                excerpt=r.chunk.text[:220],
            )
            for r in results
            if r.chunk.doc_id in answer or r.chunk.chunk_id in answer
        ]
        grounded = "INSUFFICIENT_EVIDENCE" not in answer and len(citations) > 0
        confidence = 0.88 if grounded else 0.25
        return answer, citations, confidence, grounded


class HuggingFaceLocalBackend:
    """Runs local open-weight HuggingFace LLMs (e.g., Qwen2.5 / Llama-3 / BioMistral) on GPU/CPU."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 384):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        prompt_version: str = "v2_strict_grounding",
    ) -> Tuple[str, List[Citation], float, bool]:
        self._load()
        messages = build_rag_messages(query, results, prompt_version=prompt_version)
        output = self._pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        answer = output[0]["generated_text"][-1]["content"].strip()
        citations = [
            Citation(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                section_title=r.chunk.section_title,
                excerpt=r.chunk.text[:220],
            )
            for r in results
            if r.chunk.doc_id in answer
        ]
        grounded = "INSUFFICIENT_EVIDENCE" not in answer and len(citations) > 0
        return answer, citations, 0.85 if grounded else 0.2, grounded
