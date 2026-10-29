# Changelog

## v1.4 — Camera-ready revision

- Corrected K02 and K03 model-facing lock representations in R2/R3 so that lock semantics use `LOCK/UNLOCK` rather than `ON/OFF`.
- Reran the 12 affected LLM calls (4 corrected instances × 3 strategies) and produced the definitive v1.4 result file.
- Added deterministic non-LLM baseline B0 as a post-hoc reviewer-requested comparator.
- Added independent second Gold-standard annotation and inter-annotator agreement outputs.
- Added processed CSV tables for strategy, representation, family, per-capability, and pairwise Exact Match analyses.
- Removed the redundant Hallucination Rate from the main evaluation outputs used by the revised manuscript.
- Updated scripts/notebook defaults from v1.3 to v1.4.

## v1.3 — Original submitted benchmark

- 36 source devices × 3 representations = 108 instances.
- Three prompting strategies and 324 LLM calls.
- Public source registry, prompts, experiment runner, and evaluation code.
