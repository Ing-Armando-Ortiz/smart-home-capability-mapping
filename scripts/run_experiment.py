#!/usr/bin/env python3
"""Run the capability-mapping experiment with Mistral Small 4 (mistral-small-2603)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from mistralai.client import Mistral
from pydantic import BaseModel, ConfigDict, Field

MODEL = "mistral-small-2603"
TEMPERATURE = 0.0
TOP_P = 1.0
RANDOM_SEED = 42
MAX_TOKENS = 128

CAPABILITIES = [
    "power_control",
    "brightness_control",
    "color_control",
    "color_temperature_control",
    "temperature_sensing",
    "humidity_sensing",
    "motion_detection",
    "occupancy_detection",
    "lock_control",
    "door_state_sensing",
    "temperature_setpoint_control",
    "hvac_mode_control",
    "fan_control",
    "power_sensing",
    "energy_consumption_sensing",
    "battery_level_sensing",
]

CapabilityId = Literal[
    "power_control",
    "brightness_control",
    "color_control",
    "color_temperature_control",
    "temperature_sensing",
    "humidity_sensing",
    "motion_detection",
    "occupancy_detection",
    "lock_control",
    "door_state_sensing",
    "temperature_setpoint_control",
    "hvac_mode_control",
    "fan_control",
    "power_sensing",
    "energy_consumption_sensing",
    "battery_level_sensing",
]

class CapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capabilities: list[CapabilityId] = Field(default_factory=list)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

def render_template(template: str, input_text: str) -> str:
    return template.replace("{{INPUT_TEXT}}", input_text)

def parse_json_capabilities(raw: str):
    json_valid = False
    schema_valid = False
    parsed_caps = []
    out_of_vocab = []
    try:
        obj = json.loads(raw)
        json_valid = isinstance(obj, dict)
        if json_valid and isinstance(obj.get("capabilities"), list):
            vals = obj["capabilities"]
            parsed_caps = [x for x in vals if isinstance(x, str) and x in CAPABILITIES]
            out_of_vocab = [x for x in vals if isinstance(x, str) and x not in CAPABILITIES]
            schema_valid = (
                set(obj.keys()) == {"capabilities"}
                and len(vals) == len(set(vals))
                and not out_of_vocab
                and all(isinstance(x, str) for x in vals)
            )
    except Exception:
        pass

    if not parsed_caps:
        parsed_caps = [
            c for c in CAPABILITIES
            if re.search(rf'(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])', raw)
        ]
    return sorted(set(parsed_caps)), json_valid, schema_valid, out_of_vocab

def get_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )

def make_call(client, strategy, system_prompt, user_prompt):
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        random_seed=RANDOM_SEED,
        max_tokens=MAX_TOKENS,
    )
    if strategy == "S3":
        return client.chat.parse(response_format=CapabilityResponse, **kwargs)
    return client.chat.complete(**kwargs)

def load_rows(payload_path: Path):
    with payload_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def select_dry_run(rows):
    """Select 9 non-scored plumbing calls: 3 strategies x 3 representations."""
    chosen = []
    for strategy in ("S1", "S2", "S3"):
        for rep in ("R1_NL", "R2_STRUCTURED", "R3_API_TELEMETRY"):
            chosen.append(
                next(
                    r for r in rows
                    if r["strategy_id"] == strategy and r["representation"] == rep
                )
            )
    return chosen

def run(rows, root: Path, output_path: Path, sleep_seconds: float = 0.0):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set. In Colab, add it under Secrets.")
    client = Mistral(api_key=api_key)

    prompts = {
        "S1": (
            read_text(root/"prompts/S1_system.txt"),
            read_text(root/"prompts/S1_user_template.txt")
        ),
        "S2": (
            read_text(root/"prompts/S2_system.txt"),
            read_text(root/"prompts/S2_user_template.txt")
        ),
        "S3": (
            read_text(root/"prompts/S3_system.txt"),
            read_text(root/"prompts/S3_user_template.txt")
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_order","run_id","instance_id","case_id","representation","strategy_id",
        "model_requested","model_returned","raw_response","predicted_capabilities",
        "json_valid","schema_valid","out_of_vocab","latency_ms",
        "prompt_tokens","completion_tokens","total_tokens","finish_reason",
        "retry_count","status","error"
    ]

    existing = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            existing = {r["run_id"] for r in csv.DictReader(f)}

    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for row in rows:
            if row["run_id"] in existing:
                continue

            system_prompt, user_template = prompts[row["strategy_id"]]
            user_prompt = render_template(user_template, row["input_text"])

            response = None
            error = ""
            retry_count = 0
            status = "DONE"
            start = time.perf_counter()

            for attempt in range(3):
                try:
                    response = make_call(
                        client,
                        row["strategy_id"],
                        system_prompt,
                        user_prompt,
                    )
                    retry_count = attempt
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    retry_count = attempt
                    if attempt == 2:
                        status = "API_ERROR"
                    else:
                        time.sleep(2 ** attempt)

            latency_ms = round((time.perf_counter() - start) * 1000, 3)

            if response is None:
                result = dict(
                    run_order=row["run_order"],
                    run_id=row["run_id"],
                    instance_id=row["instance_id"],
                    case_id=row["case_id"],
                    representation=row["representation"],
                    strategy_id=row["strategy_id"],
                    model_requested=MODEL,
                    model_returned="",
                    raw_response="",
                    predicted_capabilities="[]",
                    json_valid=False,
                    schema_valid=False,
                    out_of_vocab="[]",
                    latency_ms=latency_ms,
                    prompt_tokens="",
                    completion_tokens="",
                    total_tokens="",
                    finish_reason="",
                    retry_count=retry_count,
                    status=status,
                    error=error,
                )
            else:
                raw = response.choices[0].message.content
                if not isinstance(raw, str):
                    raw = json.dumps(raw, ensure_ascii=False, default=str)
                predicted, json_valid, schema_valid, oov = parse_json_capabilities(raw)
                pt, ct, tt = get_usage(response)
                result = dict(
                    run_order=row["run_order"],
                    run_id=row["run_id"],
                    instance_id=row["instance_id"],
                    case_id=row["case_id"],
                    representation=row["representation"],
                    strategy_id=row["strategy_id"],
                    model_requested=MODEL,
                    model_returned=getattr(response, "model", ""),
                    raw_response=raw,
                    predicted_capabilities=json.dumps(predicted, ensure_ascii=False),
                    json_valid=json_valid,
                    schema_valid=schema_valid,
                    out_of_vocab=json.dumps(oov, ensure_ascii=False),
                    latency_ms=latency_ms,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    finish_reason=getattr(response.choices[0], "finish_reason", ""),
                    retry_count=retry_count,
                    status=status,
                    error=error,
                )

            writer.writerow(result)
            f.flush()
            print(f'{row["run_id"]} {row["strategy_id"]} {row["instance_id"]}: {status}')
            if sleep_seconds:
                time.sleep(sleep_seconds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="data/execution_payload_v1.3.csv")
    parser.add_argument("--output", default="results/raw/mistral_small_2603.csv")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 9 plumbing calls only; do not use for paper scoring.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = load_rows(root / args.payload)
    if args.dry_run:
        rows = select_dry_run(rows)
        output = root / "results/raw/dry_run_mistral_small_2603.csv"
    else:
        output = root / args.output

    run(rows, root, output, args.sleep)

if __name__ == "__main__":
    main()
