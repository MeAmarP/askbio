"""Local-first learner progress storage for AskBio."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from askbio import PROJECT_ROOT, AskBioResponse


@dataclass(frozen=True)
class TopicProgress:
    topic: str
    attempts: int
    assessed_attempts: int
    correct_attempts: int
    average_confidence: float

    @property
    def mastery(self) -> float | None:
        if not self.assessed_attempts:
            return None
        return self.correct_attempts / self.assessed_attempts


class LearningProgressStore:
    """Store private learning activity in a local SQLite database."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PROJECT_ROOT / ".askbio" / "progress.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    correct INTEGER
                )
                """
            )

    def record(
        self,
        *,
        question: str,
        mode: str,
        level: str,
        response: AskBioResponse,
    ) -> int:
        topic = response.sources[0].label if response.sources else "Unresolved"
        scores = [source.score for source in response.sources if source.score is not None]
        confidence = sum(scores) / len(scores) if scores else 0.0
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO interactions (topic, question, mode, level, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (topic, question, mode, level, confidence),
            )
            if cursor.lastrowid is None:  # pragma: no cover - SQLite contract guard
                raise RuntimeError("SQLite did not return an interaction identifier.")
            return cursor.lastrowid

    def mark_result(self, interaction_id: int, *, correct: bool) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE interactions SET correct = ? WHERE id = ?",
                (int(correct), interaction_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown interaction: {interaction_id}")

    def topic_progress(self) -> list[TopicProgress]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    topic,
                    COUNT(*) AS attempts,
                    COUNT(correct) AS assessed_attempts,
                    COALESCE(SUM(correct), 0) AS correct_attempts,
                    AVG(confidence) AS average_confidence
                FROM interactions
                GROUP BY topic
                ORDER BY attempts DESC, topic
                """
            ).fetchall()
        return [
            TopicProgress(
                topic=row["topic"],
                attempts=row["attempts"],
                assessed_attempts=row["assessed_attempts"],
                correct_attempts=row["correct_attempts"],
                average_confidence=row["average_confidence"],
            )
            for row in rows
        ]
