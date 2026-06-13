from __future__ import annotations

import math
import re
import time
import zlib
from collections import Counter, defaultdict
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class TfidfRetriever:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self._term_freqs = [_token_counts(doc["text"]) for doc in documents]
        document_frequency: Counter[str] = Counter()
        for counts in self._term_freqs:
            document_frequency.update(counts.keys())
        total_documents = max(len(documents), 1)
        self._idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._doc_norms = [self._vector_norm(counts) for counts in self._term_freqs]

    @classmethod
    def from_papers(cls, papers: list[dict[str, Any]]) -> "TfidfRetriever":
        documents: list[dict[str, Any]] = []
        for paper in papers:
            for paragraph in paper.get("paragraphs", []):
                documents.append(
                    {
                        "paper_id": paragraph.get("paper_id") or paper.get("paper_id"),
                        "paragraph_id": paragraph.get("paragraph_id"),
                        "section": paragraph.get("section", ""),
                        "text": paragraph.get("text", ""),
                    }
                )
        return cls(documents)

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_counts = _token_counts(query)
        query_norm = self._vector_norm(query_counts)
        scored: list[dict[str, Any]] = []

        for index, document in enumerate(self.documents):
            if paper_id is not None and document.get("paper_id") != paper_id:
                continue
            score = self._cosine(query_counts, query_norm, self._term_freqs[index], self._doc_norms[index])
            scored.append({**document, "score": score})

        scored.sort(key=lambda item: (-item["score"], str(item["paragraph_id"])))
        return scored[:top_k]

    def retrieve_with_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        evidence = self.retrieve(query=query, paper_id=paper_id, top_k=top_k)
        return evidence, (time.perf_counter() - started) * 1000

    def _vector_norm(self, counts: Counter[str]) -> float:
        return math.sqrt(sum((count * self._idf.get(term, 1.0)) ** 2 for term, count in counts.items()))

    def _cosine(
        self,
        query_counts: Counter[str],
        query_norm: float,
        document_counts: Counter[str],
        document_norm: float,
    ) -> float:
        if query_norm == 0 or document_norm == 0:
            return 0.0
        dot = 0.0
        for term, query_count in query_counts.items():
            dot += query_count * self._idf.get(term, 1.0) * document_counts.get(term, 0) * self._idf.get(term, 1.0)
        return dot / (query_norm * document_norm)


class BM25Retriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._term_freqs = [_token_counts(doc["text"]) for doc in documents]
        self._doc_lengths = [sum(counts.values()) for counts in self._term_freqs]
        self._average_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_freqs:
            document_frequency.update(counts.keys())
        total_documents = max(len(documents), 1)
        self._idf = {
            term: math.log(1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_papers(cls, papers: list[dict[str, Any]]) -> "BM25Retriever":
        return cls(_documents_from_papers(papers))

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_terms = list(_token_counts(query).keys())
        scored: list[dict[str, Any]] = []
        for index, document in enumerate(self.documents):
            if paper_id is not None and document.get("paper_id") != paper_id:
                continue
            score = self._score(query_terms, index)
            scored.append({**document, "score": score, "lexical_score": score})
        scored.sort(key=lambda item: (-item["score"], str(item["paragraph_id"])))
        return scored[:top_k]

    def retrieve_with_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        evidence = self.retrieve(query=query, paper_id=paper_id, top_k=top_k)
        return evidence, (time.perf_counter() - started) * 1000

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = self._term_freqs[index]
        doc_length = self._doc_lengths[index]
        avgdl = self._average_doc_length or 1.0
        score = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * doc_length / avgdl)
            score += self._idf.get(term, 0.0) * (term_frequency * (self.k1 + 1.0)) / denominator
        return score


class HashedDenseRetriever:
    """CPU-only dense-style baseline using deterministic feature hashing."""

    def __init__(self, documents: list[dict[str, Any]], dimensions: int = 256) -> None:
        self.documents = documents
        self.dimensions = dimensions
        self._term_freqs = [_token_counts(doc["text"]) for doc in documents]
        document_frequency: Counter[str] = Counter()
        for counts in self._term_freqs:
            document_frequency.update(counts.keys())
        total_documents = max(len(documents), 1)
        self._idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._vectorize_counts(counts) for counts in self._term_freqs]
        self._norms = [_dense_norm(vector) for vector in self._vectors]

    @classmethod
    def from_papers(cls, papers: list[dict[str, Any]]) -> "HashedDenseRetriever":
        return cls(_documents_from_papers(papers))

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_vector = self._vectorize_counts(_token_counts(query))
        query_norm = _dense_norm(query_vector)
        scored: list[dict[str, Any]] = []
        for index, document in enumerate(self.documents):
            if paper_id is not None and document.get("paper_id") != paper_id:
                continue
            score = _dense_cosine(query_vector, query_norm, self._vectors[index], self._norms[index])
            scored.append({**document, "score": score, "dense_score": score})
        scored.sort(key=lambda item: (-item["score"], str(item["paragraph_id"])))
        return scored[:top_k]

    def retrieve_with_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        evidence = self.retrieve(query=query, paper_id=paper_id, top_k=top_k)
        return evidence, (time.perf_counter() - started) * 1000

    def _vectorize_counts(self, counts: Counter[str]) -> list[float]:
        vector = [0.0] * self.dimensions
        for term, count in counts.items():
            bucket = zlib.crc32(term.encode("utf-8")) % self.dimensions
            sign = -1.0 if zlib.crc32(f"{term}:sign".encode("utf-8")) % 2 else 1.0
            vector[bucket] += sign * count * self._idf.get(term, 1.0)
        return vector


def run_tfidf_baseline(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    from src.answering import answer_from_evidence

    retriever = TfidfRetriever.from_papers(papers)
    predictions: list[dict[str, Any]] = []
    for qa in qas:
        evidence, latency_ms = retriever.retrieve_with_latency(
            qa.get("question", ""),
            paper_id=qa.get("paper_id"),
            top_k=top_k,
        )
        answer = answer_from_evidence(qa.get("question", ""), evidence)
        predictions.append(
            {
                "question_id": qa.get("question_id"),
                "paper_id": qa.get("paper_id"),
                "question": qa.get("question", ""),
                "predicted_answer": answer["answer"],
                "retrieved_evidence_ids": [item["paragraph_id"] for item in evidence],
                "retrieved_evidence": evidence,
                "scores": [item["score"] for item in evidence],
                "latency_ms": latency_ms,
                "refused": answer["refused"],
            }
        )
    return predictions


def run_bm25_baseline(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return _run_retrieval_baseline(
        method="BM25-RAG",
        retriever=BM25Retriever.from_papers(papers),
        qas=qas,
        top_k=top_k,
    )


def run_dense_baseline(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return _run_retrieval_baseline(
        method="Dense-Hash-RAG",
        retriever=HashedDenseRetriever.from_papers(papers),
        qas=qas,
        top_k=top_k,
    )


def _run_retrieval_baseline(
    method: str,
    retriever: Any,
    qas: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    from src.answering import answer_from_evidence

    predictions: list[dict[str, Any]] = []
    for qa in qas:
        evidence, latency_ms = retriever.retrieve_with_latency(
            qa.get("question", ""),
            paper_id=qa.get("paper_id"),
            top_k=top_k,
        )
        answer = answer_from_evidence(qa.get("question", ""), evidence)
        predictions.append(
            {
                "method": method,
                "question_id": qa.get("question_id"),
                "paper_id": qa.get("paper_id"),
                "question": qa.get("question", ""),
                "predicted_answer": answer["answer"],
                "retrieved_evidence_ids": [item["paragraph_id"] for item in evidence],
                "retrieved_evidence": evidence,
                "scores": [item["score"] for item in evidence],
                "latency_ms": latency_ms,
                "refused": answer["refused"],
            }
        )
    return predictions


def _documents_from_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for paper in papers:
        for paragraph in paper.get("paragraphs", []):
            documents.append(
                {
                    "paper_id": paragraph.get("paper_id") or paper.get("paper_id"),
                    "paragraph_id": paragraph.get("paragraph_id"),
                    "section": paragraph.get("section", ""),
                    "text": paragraph.get("text", ""),
                }
            )
    return documents


def _dense_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dense_cosine(
    left: list[float],
    left_norm: float,
    right: list[float],
    right_norm: float,
) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(left[index] * right[index] for index in range(len(left))) / (left_norm * right_norm)


def _token_counts(text: Any) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(str(text)))
