# Deterministic baseline B0

B0 was added during camera-ready revision in response to reviewer feedback.

It uses fixed lexical and structural rules over the same 108 anonymized model-facing inputs as the LLM experiment. It uses no LLM, embeddings, external knowledge, RAG, tools, or network access at runtime.

Run:

```bash
python scripts/run_rule_baseline_v1.0.py
```

Input: `baseline_input_108_v1.0.csv`  
Output: `../results/raw/baseline_rule_based_v1.0_predictions.csv`

Because B0 is post hoc, its role is diagnostic rather than preregistered. The final rules are frozen in `scripts/run_rule_baseline_v1.0.py`.
