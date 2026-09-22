ROOT_DIR:=./
SRC_DIR:=./src
test:
	uv run pytest

check:
	uv run ruff check src tests
	uv run mypy
	uv run pytest

format:
	uv run ruff format src tests
	uv run ruff check src tests --fix

evaluate:
	uv run python src/evaluate.py

sync:
	uv sync --extra local-hf

sync-lite:
	uv sync

.PHONY: check evaluate format sync sync-lite test
