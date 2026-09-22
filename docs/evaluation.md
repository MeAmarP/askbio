# Evaluation

`evaluation/gold.json` is the active reviewed dataset. Each case contains a
question, required answer facts, and expected `filename#page` citations.

The evaluator reports:

- Recall@K: fraction of expected sources retrieved.
- Mean reciprocal rank: rank of the first expected source.
- Citation precision: fraction of returned citations that were expected.
- Answer-fact recall: fraction of required fact phrases found in the answer.
- Refusal rate and response latency.

Answer-fact recall is intentionally deterministic and inexpensive, but it does
not prove semantic correctness or groundedness. Human review or an independent
judge model should be added before using the metrics for high-stakes decisions.

The legacy `generated_questions.txt` is retained only as an experiment artifact
and is not used by the evaluator because it contains malformed and metadata-only
questions.

