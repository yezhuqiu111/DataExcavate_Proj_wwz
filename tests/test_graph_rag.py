from src.graph_rag import GraphRagRetriever, run_graph_rag
from src.retrieval import TfidfRetriever


def _sample_papers():
    return [
        {
            "paper_id": "p1",
            "paragraphs": [
                {
                    "paragraph_id": "p1::p0000",
                    "paper_id": "p1",
                    "section": "Method",
                    "text": "Graph retrieval uses entity expansion.",
                },
                {
                    "paragraph_id": "p1::p0001",
                    "paper_id": "p1",
                    "section": "Method",
                    "text": "Entity expansion improves evidence coverage.",
                },
                {
                    "paragraph_id": "p1::p0002",
                    "paper_id": "p1",
                    "section": "Background",
                    "text": "Unrelated optimizer schedule.",
                },
            ],
        },
        {
            "paper_id": "p2",
            "paragraphs": [
                {
                    "paragraph_id": "p2::p0000",
                    "paper_id": "p2",
                    "section": "Method",
                    "text": "Graph retrieval uses entity expansion for unrelated paper evidence.",
                }
            ],
        },
    ]


def test_graph_rag_expands_from_seed_terms_to_related_evidence():
    papers = _sample_papers()

    graph_retriever = GraphRagRetriever.from_papers(papers, TfidfRetriever.from_papers(papers))
    results = graph_retriever.retrieve("How does graph retrieval improve coverage?", paper_id="p1", top_k=3)

    result_ids = [item["paragraph_id"] for item in results]
    assert "p1::p0000" in result_ids
    assert "p1::p0001" in result_ids
    assert results[0]["score"] >= results[1]["score"]


def test_graph_rag_retrieve_with_trace_exposes_expansion_state():
    papers = _sample_papers()

    graph_retriever = GraphRagRetriever.from_papers(papers, TfidfRetriever.from_papers(papers))
    results, trace = graph_retriever.retrieve_with_trace(
        "How does graph retrieval improve coverage?",
        paper_id="p1",
        top_k=3,
        seed_k=1,
        graph_bonus=0.2,
    )

    result_ids = [item["paragraph_id"] for item in results]
    assert trace["graph_bonus"] == 0.2
    assert trace["returned_evidence_ids"] == result_ids
    assert trace["seed_evidence_ids"]
    assert trace["expansion_paths"]
    assert set(trace["seed_evidence_ids"]).issubset(set(trace["candidate_evidence_ids"]))
    assert all(evidence_id.startswith("p1::") for evidence_id in trace["candidate_evidence_ids"])
    assert "graph" in trace["query_terms"]
    assert "entity" in trace["expanded_terms"]
    assert "p1::p0001" in trace["candidate_evidence_ids"]


def test_run_graph_rag_includes_graph_trace_in_predictions():
    qas = [
        {
            "question_id": "q1",
            "paper_id": "p1",
            "question": "How does graph retrieval improve coverage?",
            "answers": ["Entity expansion improves evidence coverage."],
            "evidence_ids": ["p1::p0001"],
            "unanswerable": False,
        }
    ]

    predictions = run_graph_rag(_sample_papers(), qas, top_k=2)

    assert predictions[0]["graph_trace"]["returned_evidence_ids"] == predictions[0]["retrieved_evidence_ids"]
    assert predictions[0]["graph_trace"]["candidate_evidence_ids"]
