from __future__ import annotations

import pytest

from askbio import AskBioResponse, SourceCitation
from evaluate import score_case, summarize


def test_score_case_computes_retrieval_and_fact_metrics() -> None:
    case = {
        "question": "What is a cell?",
        "expected_sources": ["biology.pdf#7"],
        "expected_facts": ["basic unit", "life"],
    }
    response = AskBioResponse(
        answer="A cell is the basic unit of life.",
        sources=(
            SourceCitation(1, "Distractor", "biology.pdf", "3", 0.9),
            SourceCitation(2, "Evidence", "biology.pdf", "7", 0.8),
        ),
    )

    metrics = score_case(case, response, 0.2)

    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.citation_precision == 0.5
    assert metrics.answer_fact_recall == 1.0


def test_summarize_averages_case_results() -> None:
    case = {"question": "Q", "expected_sources": [], "expected_facts": []}
    result = score_case(case, AskBioResponse("A"), 0.4)

    summary = summarize([result])

    assert summary["cases"] == 1
    assert summary["median_latency_seconds"] == pytest.approx(0.4)
