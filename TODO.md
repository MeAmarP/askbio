# AskBio Roadmap

The product direction is an evidence-first biology tutor: answer from trusted
textbooks, show the evidence, and turn each answer into an active learning step.

## Phase 1 — Make it trustworthy

### Environment and developer experience

- [x] Add uv project metadata and select Python 3.12.
- [x] Create the local `.venv` and lock all dependencies in `uv.lock`.
- [x] Remove stale imports that prevented the project from loading.
- [x] Delay model initialization until the first UI request.
- [x] Make the PyTorch/Hugging Face inference stack an optional uv extra.
- [x] Add a lightweight remote or external-process backend for inference without PyTorch.
- [x] Add a documented authorized-textbook setup command.
- [x] Add unit and integration tests with a small fixture document.
- [x] Add linting, formatting, type checking, and CI.
- [x] Document CPU, GPU, memory, and model-download requirements.

### Retrieval and grounded answers

- [x] Replace the current double-retrieval flow with one query pass.
- [x] Correct `similarity_topk` to `similarity_top_k`.
- [x] Return structured citations with source file, page, score, and excerpt.
- [x] Display citations and evidence excerpts in the UI.
- [x] Refuse unsupported answers when retrieved evidence is insufficient.
- [x] Add document prompt-injection defenses.
- [ ] Benchmark a reranker against vector retrieval alone.

### Indexing and reliability

- [x] Resolve data paths independently of the current working directory.
- [x] Persist the vector index instead of rebuilding it at every startup.
- [x] Rebuild the index when source PDFs or indexing settings change.
- [x] Move ingestion into a dedicated command.
- [x] Add friendly setup and missing-data error states.
- [x] Log internal errors without exposing stack details to users.

### Evaluation

- [x] Replace noisy generated questions as evaluator input with a reviewed gold dataset.
- [x] Label expected facts and source pages.
- [x] Measure Recall@K, MRR, citation precision, and answer-fact recall.
- [ ] Measure groundedness, unsupported claims, latency, and memory use.
- [ ] Add quality regression thresholds and publish a baseline report.

## Phase 2 — Make it educational

### Learning modes

- [x] Add Learn, Socratic, Quiz, and Exam Review modes.
- [x] Add beginner, intermediate, and advanced explanation levels.
- [x] Generate analogies, examples, flashcards, and knowledge checks from citations.
- [x] Add progressive hints and misconception-aware feedback modes.

### Conversation and personalization

- [x] Add bounded multi-turn memory and follow-up query rewriting.
- [x] Track topics, attempts, confidence, and concept mastery in local storage.
- [ ] Recommend prerequisites and next concepts.
- [ ] Add spaced review to the local-first learner profile.
- [x] Add multilingual explanations while preserving scientific terminology.

### Learning experience

- [ ] Build a split-screen answer and textbook evidence reader.
- [ ] Add clickable follow-up prompts and related concepts.
- [ ] Add accessible loading, empty, confidence, and recovery states.
- [ ] Support keyboard navigation and screen readers.
- [ ] Export notes, flashcards, and study sessions.

## Phase 3 — Make it memorable

### Visual biology

- [ ] Extract figures with captions and page relationships.
- [ ] Retrieve diagrams alongside related passages.
- [ ] Support questions about labeled diagrams and biological processes.
- [ ] Add interactive step-through explanations for systems and cycles.
- [ ] Build a navigable concept graph with prerequisite links.

### Classroom tools

- [ ] Let teachers select chapters and create grounded assignments.
- [ ] Generate editable quizzes with citations and answer keys.
- [ ] Add class-level misconception and mastery summaries.
- [ ] Export assignments and results in common formats.
- [ ] Add role separation, retention settings, and privacy controls.

### Showcase and scale

- [ ] Publish a polished demo with a redistributable sample corpus.
- [ ] Add an evaluation dashboard comparing model and retrieval configurations.
- [ ] Support multiple textbooks and source comparison.
- [ ] Add a documented API and container deployment.
- [ ] Add quality, cost, latency, and failure observability.
- [ ] Publish an architecture diagram, benchmark report, demo video, and case study.
