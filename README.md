# LLM-Based Semantic Capability Mapping for Heterogeneous Smart-Home IoT Devices

Reproducibility repository for the preliminary empirical study:

**LLM-Based Semantic Capability Mapping for Heterogeneous Smart-Home IoT Devices: A Preliminary Empirical Study**

## Study design

The benchmark contains 36 source devices from public Zigbee2MQTT documentation. Each source case is transformed into three semantically equivalent, anonymized representations:

- `R1_NL`: natural-language technical description
- `R2_STRUCTURED`: structured/JSON-style representation
- `R3_API_TELEMETRY`: API/telemetry-style representation

This yields **108 evaluation instances**. Each instance is evaluated under three prompting strategies:

- `S1`: label-aware zero-shot
- `S2`: few-shot with four synthetic demonstrations
- `S3`: schema-constrained zero-shot

The primary experiment therefore contains **324 model calls**.

The audited Gold Standard was frozen before model execution. Vendor/model identifiers are retained only in the source registry and are not included in model-facing inputs.

## Model

Primary model: `mistral-small-2603` (Mistral Small 4, v26.03).

Configuration:

```yaml
temperature: 0.0
top_p: 1.0
random_seed: 42
max_output_tokens: 128
tools: none
web_search: false
rag: false
conversation_state: none
```

Mistral documents `mistral-small-2603` as Mistral Small 4 v26.03 with Chat Completions and Structured Outputs support.

## Repository structure

```text
data/
  benchmark_v1.3.csv
  source_registry_v1.1.csv
  execution_payload_v1.3.csv
  capabilities_v1.0.csv
prompts/
  S1_system.txt
  S1_user_template.txt
  S2_system.txt
  S2_user_template.txt
  S3_system.txt
  S3_user_template.txt
  S3_schema.json
scripts/
  run_experiment.py
  evaluate.py
notebooks/
  01_experiment_mistral_small_2603.ipynb
results/
  raw/
  processed/
experiment_config.yaml
requirements.txt
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ing-Armando-Ortiz/smart-home-capability-mapping/blob/main/notebooks/01_experiment_mistral_small_2603.ipynb)

Add a Colab Secret named `MISTRAL_API_KEY`. Never commit an API key.

Run the **9-call dry run first**. These calls validate authentication and output plumbing only and are not part of the scored experiment. After a successful dry run, execute the full 324-call experiment.

## Local execution

```bash
git clone https://github.com/Ing-Armando-Ortiz/smart-home-capability-mapping.git
cd smart-home-capability-mapping
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export MISTRAL_API_KEY="..."
python scripts/run_experiment.py --dry-run
python scripts/run_experiment.py
python scripts/evaluate.py
```

## Reproducibility safeguards

- Gold labels are not included in the execution payload.
- Vendor/model identity is removed from all model-facing inputs.
- No web search, RAG, tools, or conversation history is enabled.
- The same canonical vocabulary is supplied to S1, S2, and S3.
- S3 uses provider-enforced structured output.
- The 324-run execution order was randomized before model execution.
- API/transport failures may be retried; semantically poor answers are never retried.

## Data provenance

The source registry records the public documentation URL used for each of the 36 cases. The benchmark is a controlled research artifact derived from documented capabilities; it is not intended to exhaustively represent every function of each commercial product or the entire smart-home ecosystem.

## License

License will be finalized before the archival release.
