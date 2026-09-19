#!/usr/bin/env python3
"""Evaluate experiment outputs against benchmark Gold labels."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path

CAPABILITIES = [
    "power_control","brightness_control","color_control","color_temperature_control",
    "temperature_sensing","humidity_sensing","motion_detection","occupancy_detection",
    "lock_control","door_state_sensing","temperature_setpoint_control","hvac_mode_control",
    "fan_control","power_sensing","energy_consumption_sensing","battery_level_sensing",
]

def parse_list(s):
    if not s:
        return []
    try:
        x = json.loads(s)
        return list(x) if isinstance(x, list) else []
    except Exception:
        return [v.strip() for v in s.split(";") if v.strip()]

def load_gold(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["instance_id"]: set(v.strip() for v in r["gold_capabilities"].split(";") if v.strip())
                for r in csv.DictReader(f)}

def load_results(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def score_group(rows, gold):
    tp=fp=fn=exact=valid_json=schema_valid=0
    hallucinated=pred_total=0
    n=0
    per_cap={c:[0,0,0] for c in CAPABILITIES}
    for r in rows:
        if r["status"] != "DONE":
            continue
        n += 1
        g=gold[r["instance_id"]]
        p=set(parse_list(r["predicted_capabilities"]))
        t=len(g&p); fpp=len(p-g); fnn=len(g-p)
        tp+=t; fp+=fpp; fn+=fnn
        exact += int(p==g)
        valid_json += int(str(r["json_valid"]).lower()=="true")
        schema_valid += int(str(r["schema_valid"]).lower()=="true")
        hallucinated += fpp; pred_total += len(p)
        for c in CAPABILITIES:
            if c in g and c in p: per_cap[c][0]+=1
            if c not in g and c in p: per_cap[c][1]+=1
            if c in g and c not in p: per_cap[c][2]+=1

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    micro_f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    cap_f1=[]
    for c,(ctp,cfp,cfn) in per_cap.items():
        cp=ctp/(ctp+cfp) if ctp+cfp else 0.0
        cr=ctp/(ctp+cfn) if ctp+cfn else 0.0
        cf1=2*cp*cr/(cp+cr) if cp+cr else 0.0
        cap_f1.append(cf1)
    return {
        "n":n, "micro_precision":precision, "micro_recall":recall, "micro_f1":micro_f1,
        "macro_f1":sum(cap_f1)/len(cap_f1), "exact_match":exact/n if n else 0.0,
        "hallucination_rate":hallucinated/pred_total if pred_total else 0.0,
        "json_validity":valid_json/n if n else 0.0,
        "schema_validity":schema_valid/n if n else 0.0,
        "tp":tp,"fp":fp,"fn":fn
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark",default="data/benchmark_v1.3.csv")
    p.add_argument("--results",default="results/raw/mistral_small_2603.csv")
    p.add_argument("--output",default="results/processed/summary_metrics.csv")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    gold=load_gold(root/args.benchmark)
    rows=load_results(root/args.results)

    groups={("ALL","ALL"):rows}
    for s in ("S1","S2","S3"):
        groups[(s,"ALL")]=[r for r in rows if r["strategy_id"]==s]
        for rep in ("R1_NL","R2_STRUCTURED","R3_API_TELEMETRY"):
            groups[(s,rep)]=[r for r in rows if r["strategy_id"]==s and r["representation"]==rep]

    out=root/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    metrics=[]
    for (strategy,rep),rr in groups.items():
        m=score_group(rr,gold)
        metrics.append({"strategy":strategy,"representation":rep,**m})
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0].keys()))
        w.writeheader(); w.writerows(metrics)
    for m in metrics:
        if m["representation"]=="ALL":
            print(m)

if __name__=="__main__":
    main()
