# Architecture

```text
Authorized PDFs
      │
      ▼
Sentence splitter ──► embeddings ──► persisted vector index
                                           │
Student question ──► one retrieval pass ───┤
                                           ▼
                                 similarity filtering
                                           │
                                           ▼
                            grounded answer synthesizer
                                           │
                                           ▼
                              answer + page citations
```

## Backends

The default backend delegates generation and embeddings to Ollama. This keeps
PyTorch and model weights outside the Python virtual environment. The optional
`local-hf` extra runs Hugging Face models directly in the Python process.

## Index lifecycle

AskBio hashes every source PDF and stores those hashes with the backend,
embedding model, and chunk configuration. It loads the persisted index when the
manifest matches and rebuilds when any indexed input changes.

## Trust boundaries

- PDFs are untrusted input and may contain malicious instructions.
- Retrieved content is delimited as quoted source material in the model prompt.
- Answers are rejected when no passage survives similarity filtering.
- Internal failures are logged; the web UI returns a generic recovery message.
- Users are responsible for document licenses and processing authorization.

