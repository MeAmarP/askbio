"""Command-line operations for documents, indexing, questions, and the UI."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from askbio import AskBio, AskBioConfig, AskBioError, DocumentError

LOGGER = logging.getLogger(__name__)


def add_document(source: Path, config: AskBioConfig, *, replace: bool = False) -> Path:
    """Copy one authorized PDF into AskBio's configured document directory."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise DocumentError(f"Document not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise DocumentError("AskBio currently supports PDF documents only.")

    config.data_dir.mkdir(parents=True, exist_ok=True)
    destination = config.data_dir / source.name
    if destination.exists() and not replace:
        raise DocumentError(
            f"Document already exists: {destination}. Pass --replace to overwrite it."
        )
    shutil.copy2(source, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askbio", description="Manage and run the AskBio biology tutor."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add", help="Copy an authorized PDF into the document corpus."
    )
    add_parser.add_argument("source", type=Path)
    add_parser.add_argument("--replace", action="store_true")

    index_parser = subparsers.add_parser("index", help="Build or refresh the vector index.")
    index_parser.add_argument("--force", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask one question from the terminal.")
    ask_parser.add_argument("question")

    subparsers.add_parser("progress", help="Show locally stored topic progress.")
    subparsers.add_parser("serve", help="Launch the Gradio web interface.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    config = AskBioConfig.from_env()

    try:
        if args.command == "add":
            destination = add_document(args.source, config, replace=args.replace)
            print(f"Added {destination}")
            print("Run `uv run python src/main.py index` to build the index.")
            return 0

        if args.command == "index":
            AskBio(config, force_reindex=args.force)
            print(f"Index is ready at {config.index_dir}")
            return 0

        if args.command == "ask":
            print(AskBio(config).ask_with_sources(args.question).to_markdown())
            return 0

        if args.command == "progress":
            from progress import LearningProgressStore

            rows = LearningProgressStore().topic_progress()
            if not rows:
                print("No learning activity recorded yet.")
                return 0
            for row in rows:
                mastery = "unassessed" if row.mastery is None else f"{row.mastery:.0%}"
                print(
                    f"{row.topic}: {row.attempts} attempt(s), "
                    f"confidence {row.average_confidence:.0%}, mastery {mastery}"
                )
            return 0

        if args.command == "serve":
            from chat_interface import iface

            iface.launch()
            return 0
    except AskBioError as exc:
        LOGGER.error("%s", exc)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
