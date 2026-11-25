"""Modular vector store supporting BM25 lexical, dense semantic, and Hybrid RRF retrieval.

Uses SentenceTransformers + FAISS automatically when configured and installed, and provides
a zero-dependency reproducible TF-IDF + TruncatedSVD dense index for fast CPU/CI execution.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import DocumentChunk, RetrievalResult


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-\./%]*", text.lower())


class BM25Index:
    """Lightweight Okapi BM25 lexical retriever tailored for scientific identifiers."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freqs: Counter = Counter()
        self._doc_lens: List[int] = []
        self._term_freqs: List[Counter] = []
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def fit(self, texts: List[str]) -> None:
        self._doc_freqs = Counter()
        self._doc_lens = []
        self._term_freqs = []
        for text in texts:
            tokens = _tokenize(text)
            self._doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            for term in tf.keys():
                self._doc_freqs[term] += 1
        self._n_docs = len(texts)
        self._avgdl = sum(self._doc_lens) / max(1, self._n_docs)

    def score_all(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(self._n_docs, dtype=np.float64)
        for i in range(self._n_docs):
            dl = self._doc_lens[i]
            tf_map = self._term_freqs[i]
            s = 0.0
            for term in q_tokens:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self._doc_freqs.get(term, 0)
                idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self._avgdl))
                s += idf * (tf * (self.k1 + 1.0) / denom)
            scores[i] = s
        return scores


class HybridVectorStore:
    """Pluggable dense + lexical vector index with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        embedding_backend: str = "tfidf_svd",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        dense_dim: int = 64,
        rrf_k: int = 60,
        seed: int = 42,
    ):
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.dense_dim = dense_dim
        self.rrf_k = rrf_k
        self.seed = seed

        self.chunks: List[DocumentChunk] = []
        self._bm25 = BM25Index()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._dense_matrix: Optional[np.ndarray] = None
        self._st_model = None
        self._faiss_index = None

    def _encode_corpus(self, texts: List[str]) -> np.ndarray:
        if self.embedding_backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            if self._st_model is None:
                self._st_model = SentenceTransformer(self.embedding_model)
            vecs = np.asarray(
                self._st_model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            return vecs

        # Offline reproducible dense semantic representation (char + word TF-IDF -> SVD)
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\./%]*\b",
        )
        tfidf = self._vectorizer.fit_transform(texts)
        max_components = min(self.dense_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if max_components >= 2:
            self._svd = TruncatedSVD(n_components=max_components, random_state=self.seed)
            dense = self._svd.fit_transform(tfidf).astype(np.float32)
        else:
            self._svd = None
            dense = tfidf.toarray().astype(np.float32)

        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedding_backend == "sentence_transformers" and self._st_model is not None:
            vec = np.asarray(
                self._st_model.encode([query], normalize_embeddings=True),
                dtype=np.float32,
            )
            return vec[0]

        assert self._vectorizer is not None, "Index must be built before querying."
        q_tfidf = self._vectorizer.transform([query])
        if self._svd is not None:
            q_dense = self._svd.transform(q_tfidf).astype(np.float32)[0]
        else:
            q_dense = q_tfidf.toarray().astype(np.float32)[0]
        norm = float(np.linalg.norm(q_dense))
        return q_dense / norm if norm > 0 else q_dense

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks.")
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self._bm25.fit(texts)
        self._dense_matrix = self._encode_corpus(texts)

        # Optionally index into FAISS when installed for production inner-product search
        try:
            import faiss

            dim = self._dense_matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._dense_matrix)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def _dense_scores(self, query: str) -> np.ndarray:
        assert self._dense_matrix is not None
        q_vec = self._encode_query(query)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec.reshape(1, -1), len(self.chunks))
            out = np.zeros(len(self.chunks), dtype=np.float64)
            for idx, sc in zip(indices[0], scores[0]):
                if idx >= 0:
                    out[idx] = float(sc)
            return out
        return (self._dense_matrix @ q_vec).astype(np.float64)

    def search(self, query: str, top_k: int = 4, mode: str = "hybrid") -> List[RetrievalResult]:
        if not self.chunks:
            raise RuntimeError("Vector store is empty. Call add_chunks() first.")
        k = min(top_k, len(self.chunks))

        if mode == "bm25":
            scores = self._bm25.score_all(query)
            ranked = np.argsort(-scores)[:k]
            return [
                RetrievalResult(chunk=self.chunks[int(i)], score=float(scores[int(i)]), retrieval_mode="bm25")
                for i in ranked
            ]

        if mode == "dense":
            scores = self._dense_scores(query)
            ranked = np.argsort(-scores)[:k]
            return [
                RetrievalResult(chunk=self.chunks[int(i)], score=float(scores[int(i)]), retrieval_mode="dense")
                for i in ranked
            ]

        if mode != "hybrid":
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        bm25_scores = self._bm25.score_all(query)
        dense_scores = self._dense_scores(query)
        bm25_order = np.argsort(-bm25_scores)
        dense_order = np.argsort(-dense_scores)

        rrf_scores: Dict[int, float] = {i: 0.0 for i in range(len(self.chunks))}
        for rank, idx in enumerate(bm25_order, start=1):
            if bm25_scores[int(idx)] > 0:
                rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(dense_order, start=1):
            rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank)

        top_indices: List[Tuple[int, float]] = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]
        return [
            RetrievalResult(chunk=self.chunks[idx], score=round(score, 6), retrieval_mode="hybrid")
            for idx, score in top_indices
        ]
