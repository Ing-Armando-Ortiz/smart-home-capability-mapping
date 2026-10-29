# LLM-Based Semantic Capability Mapping for Heterogeneous Smart-Home IoT Devices

Reproducibility repository for the revised empirical study:

**LLM-Based Semantic Capability Mapping for Heterogeneous Smart-Home IoT Devices: A Preliminary Empirical Study**

This repository corresponds to the **definitive benchmark v1.4** used in the camera-ready revision.

## Study overview

The study evaluates semantic capability mapping from heterogeneous smart-home IoT descriptions into a fixed vocabulary of 16 canonical capabilities.

- 36 source devices from public Zigbee2MQTT documentation
- 6 functional device families
- 3 semantically equivalent representations per device
- 108 evaluation instances
- 3 LLM prompting strategies (324 definitive LLM calls)
- 1 deterministic non-LLM baseline (B0), added post hoc during revision as a diagnostic comparator

The three representations were **manually constructed by the researchers** from the same source evidence; no LLM was used to generate R1, R2, or R3.

- `R1_NL`: natural-language technical description
- `R2_STRUCTURED`: structured / JSON-style representation
- `R3_API_TELEMETRY`: API / telemetry-style representation

Vendor and model identifiers are excluded from model-facing inputs.

## Canonical capability vocabulary

The benchmark uses 16 canonical capability identifiers:

`power_control`, `brightness_control`, `color_control`, `color_temperature_control`, `temperature_sensing`, `humidity_sensing`, `motion_detection`, `occupancy_detection`, `lock_control`, `door_state_sensing`, `temperature_setpoint_control`, `hvac_mode_control`, `fan_control`, `power_sensing`, `energy_consumption_sensing`, and `battery_level_sensing`.

## LLM strategies

The definitive LLM experiment uses `mistral-small-2603` with:

- `S1`: label-aware zero-shot
- `S2`: few-shot with four fixed synthetic demonstrations external to the 36 benchmark devices
- `S3`: schema-constrained zero-shot

The four few-shot demonstrations were frozen before the definitive v1.4 execution and were not modified based on the reported v1.4 outcomes.

### Model configuration

```yaml
model: mistral-small-2603
temperature: 0.0
top_p: 1.0
random_seed: 42
max_output_tokens: 128
tools: none
web_search: false
rag: false
conversation_state: none
```

The 324-call execution order was randomized with seed `20260918`.

## Deterministic baseline B0

B0 is a transparent lexical/rule-based mapper operating on the **same 108 anonymized inputs** used by the LLM. It performs no network calls, uses no learned model, embeddings, RAG, tools, or external knowledge at runtime.

B0 was added **during camera-ready revision in response to reviewer feedback** and should therefore be interpreted as a post-hoc diagnostic comparator, not as a preregistered baseline.

Run it with:

```bash
python scripts/run_rule_baseline_v1.0.py
```

## Gold-standard validation

During revision, a second annotator independently re-annotated all 36 source devices using the frozen 16-capability codebook and primary device documentation. The second annotator was blinded to the original Gold labels and model/baseline predictions.

- 36 devices × 16 capabilities = **576 binary decisions**
- observed agreement: **100%**
- Cohen's kappa: **1.000**
- exact device-level capability-set agreement: **36/36**

No Gold labels were changed after the second annotation.

## Definitive v1.4 correction

Benchmark v1.4 corrects the model-facing preprocessing for lock cases **K02** and **K03** in `R2_STRUCTURED` and `R3_API_TELEMETRY`, preserving `LOCK/UNLOCK` semantics instead of the earlier incorrect `ON/OFF` rendering. The four affected instances correspond to 12 LLM calls across S1/S2/S3; these calls were rerun for the definitive v1.4 result set. Gold labels did not change.

## Main final results

| Method | Precision | Recall | Micro-F1 | Macro-F1 | Exact Match |
|---|---:|---:|---:|---:|---:|
| B0 rule-based | 0.9641 | 0.9933 | **0.9784** | **0.9688** | 89.81% |
| S1 zero-shot | 0.9690 | 0.9461 | 0.9574 | 0.9198 | 85.19% |
| S2 few-shot | **0.9764** | 0.9731 | 0.9747 | 0.9571 | **92.59%** |
| S3 schema-constrained | 0.9729 | 0.9663 | 0.9696 | 0.9454 | 90.74% |

The baseline obtains the highest aggregate Micro-F1, while the few-shot LLM obtains the highest Exact Match rate. Remaining errors are concentrated primarily in context-sensitive motion-versus-occupancy distinctions.

## Repository structure

```text
.
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE_STATUS.md
├── MANIFEST_SHA256.txt
├── .gitignore
├── experiment_config.yaml
├── requirements.txt
├── data/
│   ├── benchmark_v1.4.csv
│   ├── execution_payload_v1.4.csv
│   ├── capabilities_v1.0.csv
│   ├── source_registry_v1.1.csv
│   └── SmartHome_Capability_Mapping_Dataset_v1.4_Corrected.xlsx
├── baseline/
│   ├── README.md
│   └── baseline_input_108_v1.0.csv
├── validation/
│   ├── README.md
│   ├── Second_Annotator_Blind_Gold_Annotation_v1.0_blank.xlsx
│   ├── Second_Annotator_Blind_Gold_Annotation_v1.0_completed.xlsx
│   ├── second_annotator_decisions_v1.0.csv
│   ├── inter_annotator_agreement_summary_v1.0.csv
│   ├── inter_annotator_case_agreement_v1.0.csv
│   └── inter_annotator_per_capability_v1.0.csv
├── prompts/
│   ├── S1_system.txt
│   ├── S1_user_template.txt
│   ├── S2_system.txt
│   ├── S2_user_template.txt
│   ├── S3_system.txt
│   ├── S3_user_template.txt
│   └── S3_schema.json
├── scripts/
│   ├── run_experiment.py
│   ├── run_rule_baseline_v1.0.py
│   ├── evaluate.py
│   └── generate_report_tables.py
├── notebooks/
│   └── 01_experiment_mistral_small_2603.ipynb
└── results/
    ├── README.md
    ├── raw/
    │   ├── mistral_small_2603_v1.4.csv
    │   └── baseline_rule_based_v1.0_predictions.csv
    └── processed/
        ├── llm_summary_metrics_v1.4.csv
        ├── strategy_metrics_v1.4.csv
        ├── by_representation_v1.4.csv
        ├── by_family_v1.4.csv
        ├── per_capability_f1_v1.4.csv
        ├── pairwise_exact_match_mcnemar_v1.4.csv
        ├── Mistral_Small_2603_v1.4_Final_Analysis.xlsx
        └── B0_NonLLM_Baseline_Analysis_v1.0.xlsx
```

## Reproducing the deterministic baseline and tables

The B0 baseline and report tables require only the Python standard library:

```bash
python scripts/run_rule_baseline_v1.0.py
python scripts/generate_report_tables.py
```

The generated B0 predictions should reproduce the committed `results/raw/baseline_rule_based_v1.0_predictions.csv`.

## Re-running the LLM experiment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the API key and first run the dry run:

```bash
export MISTRAL_API_KEY="..."
python scripts/run_experiment.py --dry-run
```

Then run the definitive payload:

```bash
python scripts/run_experiment.py
python scripts/evaluate.py
python scripts/generate_report_tables.py
```

API/transport failures may be retried; semantically poor outputs are never retried.

## Reproducibility safeguards

- Gold labels are absent from the LLM execution payload.
- Vendor/model identity is removed from all model-facing inputs.
- R1/R2/R3 were manually constructed from the same source evidence.
- No web search, RAG, tools, or conversation history is enabled.
- The same canonical vocabulary is supplied to S1, S2, and S3.
- S3 uses provider-enforced structured output.
- Few-shot demonstrations are synthetic, external to the benchmark devices, and frozen for the definitive run.
- The second annotation was independent and blinded to original Gold and predictions.
- B0 is explicitly identified as a post-hoc revision comparator.

## Data provenance and scope

The source registry records the public documentation URL used for each of the 36 source devices. The benchmark is a controlled research artifact and is not intended to exhaustively represent every function of each commercial product or the full smart-home ecosystem.

The current study evaluates capability-name normalization only. It does not evaluate end-to-end device execution, hardware/network behavior, RAG, command bindings, parameter ranges, units, or agentic orchestration.

## License status

A reuse license has not yet been selected for this repository snapshot. See `LICENSE_STATUS.md` before archival publication.
