from __future__ import annotations

import pytest

from askbio import AskBioResponse, SourceCitation
from progress import LearningProgressStore


def test_progress_store_records_confidence_and_mastery(tmp_path) -> None:
    store = LearningProgressStore(tmp_path / "progress.db")
    response = AskBioResponse(
        answer="Answer",
        sources=(SourceCitation(1, "Evidence", "bio.pdf", "4", 0.8),),
    )

    interaction_id = store.record(
        question="Question",
        mode="Quiz",
        level="Intermediate",
        response=response,
    )
    store.mark_result(interaction_id, correct=True)

    progress = store.topic_progress()[0]
    assert progress.topic == "bio.pdf, page 4"
    assert progress.attempts == 1
    assert progress.average_confidence == pytest.approx(0.8)
    assert progress.mastery == 1.0


def test_unassessed_progress_has_no_mastery(tmp_path) -> None:
    store = LearningProgressStore(tmp_path / "progress.db")
    store.record(
        question="Question",
        mode="Learn",
        level="Beginner",
        response=AskBioResponse(answer="No evidence"),
    )

    assert store.topic_progress()[0].mastery is None


def test_mark_result_rejects_unknown_interaction(tmp_path) -> None:
    store = LearningProgressStore(tmp_path / "progress.db")
    with pytest.raises(KeyError, match="Unknown interaction"):
        store.mark_result(99, correct=False)
