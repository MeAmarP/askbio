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

llama-check:
	curl --fail --silent http://127.0.0.1:8080/health >/dev/null
	curl --fail --silent http://127.0.0.1:8081/health >/dev/null

sync:
	uv sync

.PHONY: check evaluate format llama-check sync test
