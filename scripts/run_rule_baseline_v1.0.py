#!/usr/bin/env python3
"""Deterministic non-LLM baseline for semantic capability mapping.

This post-hoc diagnostic baseline was added during camera-ready revision in
response to reviewer feedback. It uses only fixed lexical/structural cues over
the same anonymized 108 inputs used by the LLM experiment. It makes no network
calls and uses no external knowledge at runtime.
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

CAPABILITIES = [
    "power_control", "brightness_control", "color_control",
    "color_temperature_control", "temperature_sensing", "humidity_sensing",
    "motion_detection", "occupancy_detection", "lock_control",
    "door_state_sensing", "temperature_setpoint_control", "hvac_mode_control",
    "fan_control", "power_sensing", "energy_consumption_sensing",
    "battery_level_sensing",
]


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def map_capabilities(text: str) -> list[str]:
    t = text.lower()
    pred: set[str] = set()

    # Control / lighting
    if has(r"\bon\s*/\s*off\b|\bswitched on or off\b|get_state\(\)\s*->\s*on\|off|set_state\(on\|off\)|\bstate\s*->\s*enum\s*\{\s*on\s*\|\s*off\s*\}|\"state\"\s*:\s*\{[^{}]*\"values\"\s*:\s*\[\s*\"on\"\s*,\s*\"off\"", t):
        pred.add("power_control")
    if "brightness" in t:
        pred.add("brightness_control")
    if any(x in t for x in ("color_xy", "color_hs", "colour_xy", "colour_hs", "emitted color", "color coordinates", "colour coordinates", "rgb")):
        pred.add("color_control")
    if any(x in t for x in ("color_temp", "colour_temp", "color temperature", "colour temperature", "mired")):
        pred.add("color_temperature_control")

    # Environmental sensing
    if any(x in t for x in ("local_temperature", "measured temperature")) or has(r"(^|\n)temperature\s*->|\"temperature\"\s*:", t):
        pred.add("temperature_sensing")
    if "humidity" in t:
        pred.add("humidity_sensing")

    # Motion / presence. Deliberately lexical: middleware field 'occupancy'
    # maps to occupancy_detection even when source modality may be PIR/motion.
    if any(x in t for x in ("motion sensor", "pir", "motion_state", "motion state", "motion_detection", "motion detection")):
        pred.add("motion_detection")
    if any(x in t for x in ("occupancy", "presence", "human presence", "human-presence")):
        pred.add("occupancy_detection")

    # Locks / door state
    if any(x in t for x in ("lock/unlock", "lock_status", "lock status", "lock_state", "lock state", "unlock operation", "unlock_door")) or has(r"\"lock\"\s*:\s*\{|\block\s*->\s*enum|\{\s*lock\s*\|\s*unlock\s*\}|\[\s*\"lock\"\s*,\s*\"unlock\"\s*\]", t):
        pred.add("lock_control")
    if any(x in t for x in ("door_state", "door state", "door-state", "physical door")):
        pred.add("door_state_sensing")

    # HVAC
    if "setpoint" in t:
        pred.add("temperature_setpoint_control")
    if "system_mode" in t or "system mode" in t:
        pred.add("hvac_mode_control")
    if "fan_mode" in t or "fan mode" in t:
        pred.add("fan_control")

    # Electrical sensing
    if "instantaneous electrical power" in t or "active power" in t or has(r"(^|\n)power\s*->|\"power\"\s*:", t):
        pred.add("power_sensing")
    if "accumulated energy" in t or "energy consumption" in t or has(r"(^|\n)energy\s*->|\"energy\"\s*:", t):
        pred.add("energy_consumption_sensing")

    # Battery level (percentage/level, not a low-battery flag alone)
    if ("battery percentage" in t or "remaining battery" in t or "battery level" in t or
        has(r"\"battery\"\s*:", t) or has(r"(^|\n)battery\s*->", t)):
        pred.add("battery_level_sensing")

    return [c for c in CAPABILITIES if c in pred]


def split_caps(s: str) -> set[str]:
    return {x.strip() for x in (s or "").split(";") if x.strip()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="baseline/baseline_input_108_v1.0.csv")
    p.add_argument("--output", default="results/raw/baseline_rule_based_v1.0_predictions.csv")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = root / args.input
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(inp.open(encoding="utf-8", newline="")))
    fields = [
        "instance_id", "case_id", "representation", "device_family",
        "gold_capabilities", "baseline_capabilities", "exact_match",
        "tp", "fp", "fn",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            gold = split_caps(r["gold_capabilities"])
            pred = set(map_capabilities(r["input_text"]))
            w.writerow({
                "instance_id": r["instance_id"],
                "case_id": r["case_id"],
                "representation": r["representation"],
                "device_family": r["device_family"],
                "gold_capabilities": "; ".join(sorted(gold)),
                "baseline_capabilities": "; ".join(sorted(pred)),
                "exact_match": int(pred == gold),
                "tp": len(pred & gold),
                "fp": len(pred - gold),
                "fn": len(gold - pred),
            })

    print(f"Wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
