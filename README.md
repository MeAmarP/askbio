# AskBio

AskBio is an evidence-first biology tutor. It retrieves passages from PDFs you
are authorized to use, answers from that evidence, and shows the source file,
page, relevance score, and excerpt behind the answer.

## What is implemented

- One-pass retrieval with configurable similarity filtering.
- Explicit refusal when the corpus does not contain adequate evidence.
- Page-level citations and evidence excerpts in CLI and web answers.
- Persistent indexes that rebuild automatically when PDFs or chunk settings change.
- Lightweight Ollama inference without PyTorch in the Python environment.
- Optional Hugging Face/PyTorch inference backend.
- Safe document-ingestion and indexing commands.
- A reviewed evaluation dataset and deterministic retrieval metrics.
- Learn, Socratic, Quiz, Exam Review, Flashcards, Hint, and Feedback modes.
- Beginner, intermediate, and advanced explanation levels.
- English, Hindi, Spanish, and French responses with preserved scientific terms.
- Bounded follow-up context and local SQLite learning progress.
- Tests, Ruff linting/formatting, mypy, coverage, and GitHub Actions CI.

See [TODO.md](TODO.md) for the product roadmap and current progress.

## Requirements

- Python 3.12 (managed automatically by uv).
- About 650 MB for the default Python environment, including development tools.
- [Ollama](https://ollama.com/) running locally or at a configured URL.
- About 2.5 GB for the default Ollama language and embedding models.
- One or more PDF documents that you have permission to process.

The optional Hugging Face backend is substantially larger because it installs
PyTorch and GPU runtime packages. CPU inference is supported but will be much
slower than Ollama or a compatible GPU.

## Setup

Install the lightweight environment:

```bash
uv sync
```

Install Ollama separately, start it, and fetch the default models:

```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

Copy the example configuration if you want to customize paths or models:

```bash
cp .env.example .env
```

### Optional Hugging Face backend

```bash
uv sync --extra local-hf
ASKBIO_BACKEND=huggingface uv run python src/main.py serve
```

This installs PyTorch, Transformers, sentence-transformers, bitsandbytes, and
the Hugging Face LlamaIndex integrations.

## Add authorized documents

AskBio deliberately does not download or redistribute a textbook. Add a PDF
that you are legally permitted to process:

```bash
uv run python src/main.py add /path/to/biology-textbook.pdf
uv run python src/main.py index
```

The project was originally demonstrated with OpenStax *Concepts of Biology*.
Its current web edition includes specific attribution and AI-ingestion terms;
review those terms and obtain any necessary permission before adding that
content. The source page is
[OpenStax Concepts of Biology](https://openstax.org/books/concepts-biology/pages/1-introduction).

## Run AskBio

Ask one question:

```bash
uv run python src/main.py ask "What is inductive reasoning?"
```

Launch the web interface:

```bash
uv run python src/main.py serve
```

Review locally stored learning progress:

```bash
uv run python src/main.py progress
```

The first index build calls the embedding backend. Later starts reuse the
persisted `.askbio_index` until documents or indexing settings change.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASKBIO_BACKEND` | `ollama` | `ollama` or `huggingface` |
| `ASKBIO_DATA_DIR` | `data/sample` | Recursive PDF corpus |
| `ASKBIO_INDEX_DIR` | `.askbio_index` | Persistent vector index |
| `ASKBIO_LLM_MODEL` | `phi3:mini` | Answer-generation model |
| `ASKBIO_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `ASKBIO_TOP_K` | `4` | Retrieved candidates |
| `ASKBIO_SIMILARITY_CUTOFF` | `0.55` | Minimum accepted similarity |
| `ASKBIO_CHUNK_SIZE` | `512` | Text chunk size |
| `ASKBIO_CHUNK_OVERLAP` | `64` | Text overlap between chunks |

## Quality checks

```bash
make check
```

Or run each check directly:

```bash
uv run ruff check src tests
uv run mypy
uv run pytest --cov=src
```

## Evaluation

The default gold set lives in `evaluation/gold.json`. It replaces the original
unfiltered generated-question dump as the evaluator input and contains reviewed
facts and source pages.

With the matching document corpus indexed:

```bash
make evaluate
```

The JSON report records Recall@K, mean reciprocal rank, citation precision,
answer-fact recall, refusal rate, and latency. Reports are written under
`evaluation/reports/` and should be reviewed before setting release thresholds.

## Current limitations

- An end-to-end model run requires an authorized corpus and a running Ollama
  server, neither of which is bundled with the repository.
- Progress mastery is recorded through the storage API, but the UI does not yet
  expose answer self-assessment controls.
- Diagram retrieval and classroom tooling are scheduled for Phase 3.
- Prompt-injection resistance is defense-in-depth, not a security guarantee.

## License

AskBio's source code is licensed under GPL-3.0. Source documents retain their
own licenses and are not covered by the repository license.
