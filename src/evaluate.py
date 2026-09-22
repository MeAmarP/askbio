"""Deterministic retrieval and answer-quality evaluation for AskBio."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from askbio import PROJECT_ROOT, AskBio, AskBioResponse


@dataclass(frozen=True)
class CaseMetrics:
    question: str
    recall_at_k: float
    reciprocal_rank: float
    citation_precision: float
    answer_fact_recall: float
    latency_seconds: float
    insufficient_evidence: bool


def citation_id(file_name: str, page: str | None) -> str:
    return f"{file_name}#{page or ''}"


def score_case(
    case: dict[str, Any], response: AskBioResponse, latency_seconds: float
) -> CaseMetrics:
    expected_sources = set(case.get("expected_sources", []))
    retrieved = [citation_id(source.file_name, source.page) for source in response.sources]
    relevant_positions = [
        position
        for position, source_id in enumerate(retrieved, start=1)
        if source_id in expected_sources
    ]
    relevant_retrieved = len(set(retrieved) & expected_sources)
    recall = relevant_retrieved / len(expected_sources) if expected_sources else 1.0
    precision = relevant_retrieved / len(retrieved) if retrieved else float(not expected_sources)
    reciprocal_rank = 1.0 / relevant_positions[0] if relevant_positions else 0.0

    normalized_answer = response.answer.casefold()
    facts = [str(fact).casefold() for fact in case.get("expected_facts", [])]
    fact_recall = sum(fact in normalized_answer for fact in facts) / len(facts) if facts else 1.0
    return CaseMetrics(
        question=case["question"],
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        citation_precision=precision,
        answer_fact_recall=fact_recall,
        latency_seconds=latency_seconds,
        insufficient_evidence=response.insufficient_evidence,
    )


def summarize(results: list[CaseMetrics]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "cases": float(len(results)),
        "mean_recall_at_k": statistics.fmean(result.recall_at_k for result in results),
        "mean_reciprocal_rank": statistics.fmean(result.reciprocal_rank for result in results),
        "mean_citation_precision": statistics.fmean(
            result.citation_precision for result in results
        ),
        "mean_answer_fact_recall": statistics.fmean(
            result.answer_fact_recall for result in results
        ),
        "median_latency_seconds": statistics.median(result.latency_seconds for result in results),
        "insufficient_evidence_rate": statistics.fmean(
            float(result.insufficient_evidence) for result in results
        ),
    }


def run(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))["cases"]
    assistant = AskBio()
    results = []
    for case in cases:
        started = time.perf_counter()
        response = assistant.ask_with_sources(case["question"])
        results.append(score_case(case, response, time.perf_counter() - started))

    report = {
        "dataset": str(dataset_path),
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "gold.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "reports" / "latest.json",
    )
    args = parser.parse_args()
    report = run(args.dataset, args.output)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
