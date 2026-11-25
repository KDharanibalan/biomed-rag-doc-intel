"""Domain-aware chunking strategies for biomedical SOPs, SDSs, COAs, and studies."""

import re
from pathlib import Path
from typing import List, Tuple

from .schemas import DocumentChunk


def infer_doc_metadata(text: str, filename: str) -> Tuple[str, str]:
    """Extract canonical document ID and type (SOP/SDS/COA/STUDY) from header metadata."""
    match = re.search(r"\*\*Document ID:\*\*\s*([A-Za-z0-9\-_]+)", text)
    doc_id = match.group(1).strip() if match else Path(filename).stem.upper()

    upper_name = f"{doc_id}_{filename}".upper()
    if "SOP" in upper_name:
        doc_type = "SOP"
    elif "SDS" in upper_name:
        doc_type = "SDS"
    elif "COA" in upper_name:
        doc_type = "COA"
    else:
        doc_type = "STUDY"
    return doc_id, doc_type


def _split_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []
    step = max(1, chunk_size - overlap)
    segments = []
    for start in range(0, len(words), step):
        seg = " ".join(words[start : start + chunk_size]).strip()
        if seg:
            segments.append(seg)
        if start + chunk_size >= len(words):
            break
    return segments


def chunk_document(
    text: str,
    source_file: str,
    strategy: str = "section_aware",
    chunk_size_words: int = 140,
    overlap_words: int = 30,
) -> List[DocumentChunk]:
    """Split a markdown/text biological document into provenance-tagged chunks.

    Supports three strategies for controlled ablation experiments:
    - ``fixed``: non-overlapping fixed word windows.
    - ``sliding_window``: overlapping word windows ignoring section headings.
    - ``section_aware``: splits along markdown section boundaries while preserving
      header metadata and subdividing long sections with overlap.
    """
    doc_id, doc_type = infer_doc_metadata(text, source_file)
    chunks: List[DocumentChunk] = []

    if strategy == "fixed":
        segments = _split_words(text, chunk_size_words, overlap=0)
        for idx, seg in enumerate(segments):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}::fixed_{idx:02d}",
                    doc_id=doc_id,
                    source_file=source_file,
                    doc_type=doc_type,
                    section_title=f"Window {idx + 1}",
                    text=seg,
                    token_count=len(seg.split()),
                )
            )
        return chunks

    if strategy == "sliding_window":
        segments = _split_words(text, chunk_size_words, overlap=overlap_words)
        for idx, seg in enumerate(segments):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}::win_{idx:02d}",
                    doc_id=doc_id,
                    source_file=source_file,
                    doc_type=doc_type,
                    section_title=f"Sliding Window {idx + 1}",
                    text=seg,
                    token_count=len(seg.split()),
                )
            )
        return chunks

    if strategy != "section_aware":
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    # Section-aware splitting preserving document title/metadata context
    lines = text.splitlines()
    doc_title = ""
    current_section = "Header & Metadata"
    buffer: List[str] = []
    sections: List[Tuple[str, str]] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not doc_title:
            doc_title = stripped.lstrip("# ").strip()
            buffer.append(stripped)
        elif stripped.startswith("## "):
            if buffer:
                sections.append((current_section, "\n".join(buffer).strip()))
            current_section = stripped.lstrip("# ").strip()
            buffer = [f"Document: {doc_title} ({doc_id}) | Section: {current_section}"]
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_section, "\n".join(buffer).strip()))

    chunk_idx = 0
    for sec_title, sec_body in sections:
        if not sec_body.strip():
            continue
        sub_segments = _split_words(sec_body, chunk_size_words, overlap=overlap_words)
        for seg in sub_segments:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}::sec_{chunk_idx:02d}",
                    doc_id=doc_id,
                    source_file=source_file,
                    doc_type=doc_type,
                    section_title=sec_title,
                    text=seg,
                    token_count=len(seg.split()),
                )
            )
            chunk_idx += 1

    return chunks


def load_corpus_directory(
    corpus_dir: Path,
    strategy: str = "section_aware",
    chunk_size_words: int = 140,
    overlap_words: int = 30,
) -> List[DocumentChunk]:
    """Ingest all markdown/text files from a corpus directory."""
    all_chunks: List[DocumentChunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        all_chunks.extend(
            chunk_document(
                raw,
                source_file=path.name,
                strategy=strategy,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )
        )
    return all_chunks
