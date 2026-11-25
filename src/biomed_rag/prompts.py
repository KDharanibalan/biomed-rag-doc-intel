"""Versioned prompt templates with strict citation grounding and schema enforcement."""

from typing import List
from .schemas import RetrievalResult

PROMPT_VERSIONS = {
    "v1_basic_qa": (
        "You are a biomedical document analyst. Answer the question using the provided context. "
        "Include document IDs where relevant."
    ),
    "v2_strict_grounding": (
        "You are a precision biomedical and regulatory document intelligence assistant. "
        "Answer ONLY using verifiable facts explicitly present in the retrieved context blocks. "
        "Rules:\n"
        "1. Every factual claim must cite its source chunk as [DOC_ID | CHUNK_ID].\n"
        "2. Preserve exact numerical specifications, temperatures, units, and compliance codes.\n"
        "3. If the context does not contain sufficient evidence, respond with: "
        "'INSUFFICIENT_EVIDENCE: The indexed documents do not specify this information.'\n"
        "4. Never extrapolate or fabricate experimental parameters."
    ),
    "v2_structured_extraction": (
        "Extract a strict JSON object from the provided biomedical/regulatory document context with keys: "
        "'doc_id', 'doc_type', 'title', 'key_fields' (dict of parameter->value), and "
        "'safety_or_qc_requirements' (list of strings). Only extract explicitly stated fields."
    ),
}


def format_context_block(results: List[RetrievalResult]) -> str:
    """Serialize retrieved chunks with explicit provenance tags for LLM grounding."""
    blocks = []
    for idx, res in enumerate(results, start=1):
        c = res.chunk
        blocks.append(
            f"[Context {idx} | doc_id={c.doc_id} | chunk_id={c.chunk_id} | "
            f"section={c.section_title} | score={res.score:.4f}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_rag_messages(
    query: str,
    results: List[RetrievalResult],
    prompt_version: str = "v2_strict_grounding",
) -> List[dict]:
    """Assemble chat-formatted system + user prompts for OpenAI/vLLM/HuggingFace backends."""
    system_prompt = PROMPT_VERSIONS.get(prompt_version, PROMPT_VERSIONS["v2_strict_grounding"])
    context = format_context_block(results)
    user_prompt = (
        f"Retrieved Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a concise, strictly grounded answer with inline citations."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
