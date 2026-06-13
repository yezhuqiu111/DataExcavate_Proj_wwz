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
EXPANSION_MATCH_WEIGHT = 0.25
MAX_EXPANSION_MATCHES = 2
MAX_EXPANSION_TERMS = 120
MAX_NEIGHBORS_PER_TERM = 12
MAX_SEED_TERMS_FOR_EXPANSION = 8
MAX_TERM_DF_RATIO = 0.2


class GraphRagRetriever:
    def __init__(self, papers: list[dict[str, Any]], tfidf: TfidfRetriever) -> None:
        self.tfidf = tfidf
        self.documents = {doc["paragraph_id"]: doc for doc in tfidf.documents}
        self.paragraph_terms: dict[str, set[str]] = {}
        self.term_paragraphs: dict[str, set[str]] = defaultdict(set)
        self.term_document_frequency: dict[str, int] = {}
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
        use_edges: bool = True,
    ) -> list[dict[str, Any]]:
        evidence, _trace = self.retrieve_with_trace(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
            use_edges=use_edges,
        )
        return evidence

    def retrieve_with_trace(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
        use_edges: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        seeds = self.tfidf.retrieve(query, paper_id=paper_id, top_k=seed_k)
        seed_ids = {seed["paragraph_id"] for seed in seeds}
        query_terms = extract_terms(query)
        if use_edges:
            expansion_terms, expansion_paths = self._collect_expansion_terms(seeds, query_terms)
        else:
            expansion_terms = set(query_terms)
            expansion_paths = []

        candidate_ids = set(seed_ids)
        for term in expansion_terms:
            candidate_ids.update(self.term_paragraphs.get(term, set()))

        lexical_scores = {
            item["paragraph_id"]: item["score"]
            for item in self.tfidf.retrieve(query, paper_id=paper_id, top_k=len(self.tfidf.documents))
        }
        scored: list[dict[str, Any]] = []
        filtered_candidate_ids: set[str] = set()
        expanded_only_terms = expansion_terms - query_terms
        for paragraph_id in candidate_ids:
            document = self.documents.get(paragraph_id)
            if not document or (paper_id is not None and document.get("paper_id") != paper_id):
                continue
            filtered_candidate_ids.add(paragraph_id)
            paragraph_terms = self.paragraph_terms.get(paragraph_id, set())
            query_matches = len(query_terms & paragraph_terms)
            expansion_matches = len(expanded_only_terms & paragraph_terms)
            graph_score = query_matches + EXPANSION_MATCH_WEIGHT * min(expansion_matches, MAX_EXPANSION_MATCHES)
            lexical_score = lexical_scores.get(paragraph_id, 0.0)
            score = lexical_score + graph_bonus * graph_score
            scored.append(
                {
                    **document,
                    "score": score,
                    "lexical_score": lexical_score,
                    "graph_matches": query_matches,
                    "expansion_matches": expansion_matches,
                    "graph_score": graph_score,
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
            "expansion_match_weight": EXPANSION_MATCH_WEIGHT,
            "max_expansion_matches": MAX_EXPANSION_MATCHES,
            "use_edges": use_edges,
        }
        return evidence, trace

    def retrieve_with_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
        use_edges: bool = True,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        evidence = self.retrieve(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
            use_edges=use_edges,
        )
        return evidence, (time.perf_counter() - started) * 1000

    def retrieve_with_trace_and_latency(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int = 5,
        seed_k: int = 2,
        graph_bonus: float = 0.15,
        use_edges: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        started = time.perf_counter()
        evidence, trace = self.retrieve_with_trace(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            seed_k=seed_k,
            graph_bonus=graph_bonus,
            use_edges=use_edges,
        )
        return evidence, trace, (time.perf_counter() - started) * 1000

    def has_query_match(self, query: str, evidence: list[dict[str, Any]]) -> bool:
        query_terms = extract_terms(query)
        return any(query_terms & self.paragraph_terms.get(item.get("paragraph_id"), set()) for item in evidence)

    def query_term_overlap(self, query: str, evidence: list[dict[str, Any]]) -> int:
        query_terms = extract_terms(query)
        if not query_terms or not evidence:
            return 0
        top_terms = self.paragraph_terms.get(evidence[0].get("paragraph_id"), set())
        return len(query_terms & top_terms)

    def _is_expandable_term(self, term: str) -> bool:
        total_paragraphs = max(len(self.paragraph_terms), 1)
        frequency = self.term_document_frequency.get(term, 0)
        allowed = max(2, int(total_paragraphs * MAX_TERM_DF_RATIO))
        return frequency <= allowed

    def _collect_expansion_terms(
        self,
        seeds: list[dict[str, Any]],
        query_terms: set[str],
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Expand only from query terms and a small set of low-frequency seed terms."""
        expansion_terms = set(query_terms)
        expansion_paths: list[dict[str, Any]] = []
        expanded_only_budget = MAX_EXPANSION_TERMS

        for seed in seeds:
            seed_terms = self.paragraph_terms.get(seed["paragraph_id"], set())
            overlapping_seed_terms = sorted(seed_terms & query_terms)
            other_seed_terms = sorted(seed_terms - query_terms)[:MAX_SEED_TERMS_FOR_EXPANSION]
            source_terms = list(query_terms) + overlapping_seed_terms + other_seed_terms
            seen_sources: set[str] = set()
            for term in source_terms:
                if term in seen_sources or not self._is_expandable_term(term):
                    continue
                seen_sources.add(term)
                neighbors = sorted(
                    neighbor
                    for neighbor in self.neighbors.get(term, set())
                    if neighbor != term and self._is_expandable_term(neighbor)
                )[:MAX_NEIGHBORS_PER_TERM]
                for neighbor in neighbors:
                    if neighbor in query_terms:
                        continue
                    if len(expansion_terms - query_terms) >= expanded_only_budget:
                        return expansion_terms, expansion_paths
                    if neighbor not in expansion_terms:
                        expansion_terms.add(neighbor)
                        expansion_paths.append(
                            {
                                "seed_evidence_id": seed["paragraph_id"],
                                "source_term": term,
                                "expanded_term": neighbor,
                            }
                        )
        return expansion_terms, expansion_paths

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
        self.term_document_frequency = {
            term: len(paragraph_ids) for term, paragraph_ids in self.term_paragraphs.items()
        }


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
    use_edges: bool = True,
    enable_refusal: bool = True,
    method: str = "GraphRAG",
) -> list[dict[str, Any]]:
    retriever = GraphRagRetriever.from_papers(papers)
    predictions: list[dict[str, Any]] = []
    for qa in qas:
        evidence, trace, latency_ms = retriever.retrieve_with_trace_and_latency(
            qa.get("question", ""),
            paper_id=qa.get("paper_id"),
            top_k=top_k,
            use_edges=use_edges,
        )
        graph_match = retriever.has_query_match(qa.get("question", ""), evidence)
        query_overlap = retriever.query_term_overlap(qa.get("question", ""), evidence)
        answer = answer_from_evidence(
            qa.get("question", ""),
            evidence,
            has_graph_match=graph_match,
            query_term_overlap=query_overlap,
            enable_refusal=enable_refusal,
        )
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
                "graph_match": graph_match,
                "graph_trace": trace,
                "use_edges": use_edges,
                "enable_refusal": enable_refusal,
            }
        )
    return predictions
