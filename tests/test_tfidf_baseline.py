from src.answering import answer_from_evidence
from src.evaluate import evaluate_predictions
from src.retrieval import BM25Retriever, HashedDenseRetriever, TfidfRetriever


def test_tfidf_baseline_retrieves_answers_and_scores_predictions():
    papers = [
        {
            "paper_id": "p1",
            "paragraphs": [
                {
                    "paragraph_id": "p1::p0000",
                    "paper_id": "p1",
                    "section": "Intro",
                    "text": "Graph retrieval connects evidence terms for question answering.",
                },
                {
                    "paragraph_id": "p1::p0001",
                    "paper_id": "p1",
                    "section": "Intro",
                    "text": "The training loop uses unrelated optimizer details.",
                },
            ],
        }
    ]
    qas = [
        {
            "question_id": "p1::q1",
            "paper_id": "p1",
            "question": "What connects evidence terms?",
            "answers": ["Graph retrieval"],
            "evidence_ids": ["p1::p0000"],
            "unanswerable": False,
        }
    ]

    retriever = TfidfRetriever.from_papers(papers)
    evidence = retriever.retrieve(qas[0]["question"], paper_id="p1", top_k=2)
    answer = answer_from_evidence(qas[0]["question"], evidence)
    prediction = {
        "question_id": qas[0]["question_id"],
        "predicted_answer": answer["answer"],
        "retrieved_evidence_ids": [item["paragraph_id"] for item in evidence],
        "retrieved_evidence": evidence,
        "scores": [item["score"] for item in evidence],
        "latency_ms": 1.0,
        "refused": answer["refused"],
    }
    metrics = evaluate_predictions(qas, [prediction], top_k=5)

    assert evidence[0]["paragraph_id"] == "p1::p0000"
    assert answer["answer"] == "Graph retrieval connects evidence terms for question answering."
    assert metrics["evidence_recall_at_5"] == 1.0
    assert metrics["evidence_recall_at_k"] == 1.0
    assert metrics["evidence_precision_at_k"] == 0.5
    assert metrics["evidence_f1_at_k"] > 0
    assert metrics["answer_exact_match"] == 0.0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["top_k"] == 5
    assert metrics["answer_token_f1"] > 0
    assert metrics["average_latency_ms"] == 1.0


def test_metrics_name_requested_top_k_explicitly():
    qas = [
        {
            "question_id": "q1",
            "answers": ["answer"],
            "evidence_ids": ["p1"],
            "unanswerable": False,
        }
    ]
    predictions = [
        {
            "question_id": "q1",
            "predicted_answer": "answer",
            "retrieved_evidence_ids": ["p1"],
            "latency_ms": 1.0,
            "refused": False,
        }
    ]

    metrics = evaluate_predictions(qas, predictions, top_k=1)

    assert metrics["top_k"] == 1
    assert metrics["evidence_recall_at_k"] == 1.0
    assert metrics["evidence_recall_at_1"] == 1.0
    assert metrics["evidence_recall_at_5"] is None


def test_bm25_and_dense_retrievers_rank_relevant_paragraphs():
    papers = [
        {
            "paper_id": "p1",
            "paragraphs": [
                {
                    "paragraph_id": "p1::p0000",
                    "paper_id": "p1",
                    "section": "Intro",
                    "text": "Graph retrieval connects evidence terms for question answering.",
                },
                {
                    "paragraph_id": "p1::p0001",
                    "paper_id": "p1",
                    "section": "Intro",
                    "text": "The training loop uses unrelated optimizer details.",
                },
            ],
        }
    ]

    bm25 = BM25Retriever.from_papers(papers)
    dense = HashedDenseRetriever.from_papers(papers)

    assert bm25.retrieve("What connects evidence terms?", paper_id="p1")[0]["paragraph_id"] == "p1::p0000"
    assert dense.retrieve("What connects evidence terms?", paper_id="p1")[0]["paragraph_id"] == "p1::p0000"
