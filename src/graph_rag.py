from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from src.answering import answer_from_evidence
from src.retrieval import TOKEN_RE, TfidfRetriever


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "uses",
    "what",
    "when",
    "where",
    "which",
    "with",
    "also",
    "been",
    "could",
    "does",
    "have",
    "into",
    "may",
    "more",
    "most",
    "paper",
    "should",
    "such",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "using",
    "were",
    "will",
    "would",
}

TRACE_SAMPLE_LIMIT = 100


class GraphRagRetriever:
    def __init__(self, papers: list[dict[str, Any]], tfidf: TfidfRetriever) -> None:
        self.tfidf = tfidf
        self.documents = {doc["paragraph_id"]: doc for doc in tfidf.documents}
        self.paragraph_terms: dict[str, set[str]] = {}
        self.term_paragraphs: dict[str, set[str]] = defaultdict(set)
        self.neighbors: dict[str, set[str]] = defaultdict(set)
        self._build_graph(papers)

    @classmethod
    def from_papers(cls, papers: list[dict[str, Any]], tfidf: TfidfRetriever | None = None) -> "GraphRagRetriever":
        return cls(papers, tfidf or TfidfRetriever.from_papers(papers))

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
    ) -> list[dict[str, Any]]:
        evidence, _trace = self.retrieve_with_trace(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
        )
        return evidence

    def retrieve_with_trace(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        seeds = self.tfidf.retrieve(query, paper_id=paper_id, top_k=seed_k)
        seed_ids = {seed["paragraph_id"] for seed in seeds}
        query_terms = extract_terms(query)
        expansion_terms = set(query_terms)
        expansion_paths: list[dict[str, Any]] = []
        for seed in seeds:
            seed_terms = self.paragraph_terms.get(seed["paragraph_id"], set())
            expansion_terms.update(seed_terms)
            for term in seed_terms | query_terms:
                neighbors = self.neighbors.get(term, set())
                expansion_terms.update(neighbors)
                for neighbor in sorted(neighbors - {term}):
                    expansion_paths.append(
                        {
                            "seed_evidence_id": seed["paragraph_id"],
                            "source_term": term,
                            "expanded_term": neighbor,
                        }
                    )

        candidate_ids = set(seed_ids)
        for term in expansion_terms:
            candidate_ids.update(self.term_paragraphs.get(term, set()))

        lexical_scores = {
            item["paragraph_id"]: item["score"]
            for item in self.tfidf.retrieve(query, paper_id=paper_id, top_k=len(self.tfidf.documents))
        }
        scored: list[dict[str, Any]] = []
        filtered_candidate_ids: set[str] = set()
        for paragraph_id in candidate_ids:
            document = self.documents.get(paragraph_id)
            if not document or (paper_id is not None and document.get("paper_id") != paper_id):
                continue
            filtered_candidate_ids.add(paragraph_id)
            graph_matches = len(query_terms & self.paragraph_terms.get(paragraph_id, set()))
            if paragraph_id not in seed_ids:
                graph_matches += 1
            lexical_score = lexical_scores.get(paragraph_id, 0.0)
            score = lexical_score + graph_bonus * graph_matches
            scored.append(
                {
                    **document,
                    "score": score,
                    "lexical_score": lexical_score,
                    "graph_matches": graph_matches,
                }
            )

        scored.sort(key=lambda item: (-item["score"], str(item["paragraph_id"])))
        evidence = scored[:top_k]
        expanded_terms = sorted(expansion_terms - query_terms)
        candidate_evidence_ids = sorted(filtered_candidate_ids)
        trace = {
            "seed_evidence_ids": sorted(seed_ids),
            "query_terms": sorted(query_terms),
            "expanded_terms": expanded_terms[:TRACE_SAMPLE_LIMIT],
            "expanded_terms_count": len(expanded_terms),
            "expansion_paths": expansion_paths[:50],
            "candidate_evidence_ids": candidate_evidence_ids[:TRACE_SAMPLE_LIMIT],
            "candidate_evidence_count": len(candidate_evidence_ids),
            "returned_evidence_ids": [item["paragraph_id"] for item in evidence],
            "graph_bonus": graph_bonus,
        }
        return evidence, trace

    def retrieve_with_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        evidence = self.retrieve(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
        )
        return evidence, (time.perf_counter() - started) * 1000

    def retrieve_with_trace_and_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        started = time.perf_counter()
        evidence, trace = self.retrieve_with_trace(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
        )
        return evidence, trace, (time.perf_counter() - started) * 1000

    def has_query_match(self, query: str, evidence: list[dict[str, Any]]) -> bool:
        query_terms = extract_terms(query)
        return any(query_terms & self.paragraph_terms.get(item.get("paragraph_id"), set()) for item in evidence)

    def _build_graph(self, papers: list[dict[str, Any]]) -> None:
        for paper in papers:
            for paragraph in paper.get("paragraphs", []):
                paragraph_id = paragraph.get("paragraph_id")
                terms = extract_terms(paragraph.get("text", ""))
                if not paragraph_id or not terms:
                    continue
                self.paragraph_terms[paragraph_id] = terms
                for term in terms:
                    self.term_paragraphs[term].add(paragraph_id)
                for term in terms:
                    self.neighbors[term].update(terms - {term})


def extract_terms(text: Any) -> set[str]:
    terms = {token.lower() for token in TOKEN_RE.findall(str(text)) if len(token) >= 4}
    return {
        term
        for term in terms
        if term not in STOPWORDS
        and not term.isdigit()
        and not term.startswith(("bibref", "figref", "tabref"))
    }


def run_graph_rag(
    papers: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    retriever = GraphRagRetriever.from_papers(papers)
    predictions: list[dict[str, Any]] = []
    for qa in qas:
        evidence, trace, latency_ms = retriever.retrieve_with_trace_and_latency(
            qa.get("question", ""),
            paper_id=qa.get("paper_id"),
            top_k=top_k,
        )
        graph_match = retriever.has_query_match(qa.get("question", ""), evidence)
        answer = answer_from_evidence(qa.get("question", ""), evidence, has_graph_match=graph_match)
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
                "graph_match": graph_match,
                "graph_trace": trace,
            }
        )
    return predictions
