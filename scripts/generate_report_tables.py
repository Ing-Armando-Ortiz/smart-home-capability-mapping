#!/usr/bin/env python3
"""Generate compact CSV tables reported in the revised paper from final v1.4 outputs."""

from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path

CAPS=[
"power_control","brightness_control","color_control","color_temperature_control",
"temperature_sensing","humidity_sensing","motion_detection","occupancy_detection",
"lock_control","door_state_sensing","temperature_setpoint_control","hvac_mode_control",
"fan_control","power_sensing","energy_consumption_sensing","battery_level_sensing"]


def split_semicolon(s): return {x.strip() for x in (s or '').split(';') if x.strip()}
def parse_json_list(s):
    try:
        v=json.loads(s); return set(v) if isinstance(v,list) else set()
    except Exception: return split_semicolon(s)

def f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    return 2*p*r/(p+r) if p+r else 0.0

def score(pred_by_inst,gold,instances):
    tp=fp=fn=exact=0; pcs=[]
    for iid in instances:
        g=gold[iid]; p=pred_by_inst[iid]
        tp+=len(g&p); fp+=len(p-g); fn+=len(g-p); exact+=int(g==p)
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/(tp+fn) if tp+fn else 0.0
    micro=2*prec*rec/(prec+rec) if prec+rec else 0.0
    for c in CAPS:
        ctp=cfp=cfn=0
        for iid in instances:
            g=gold[iid]; p=pred_by_inst[iid]
            ctp+=int(c in g and c in p); cfp+=int(c not in g and c in p); cfn+=int(c in g and c not in p)
        pcs.append(f1(ctp,cfp,cfn))
    return {'n':len(instances),'precision':prec,'recall':rec,'micro_f1':micro,'macro_f1':sum(pcs)/len(pcs),'exact_match':exact/len(instances),'tp':tp,'fp':fp,'fn':fn}

def exact_mcnemar_p(b,c):
    # Exact two-sided binomial test under p=0.5 for discordant pairs.
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    tail=sum(math.comb(n,i) for i in range(0,k+1))/(2**n)
    return min(1.0,2*tail)

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    root=Path(__file__).resolve().parents[1]
    bench=list(csv.DictReader((root/'data/benchmark_v1.4.csv').open(encoding='utf-8',newline='')))
    gold={r['instance_id']:split_semicolon(r['gold_capabilities']) for r in bench}
    meta={r['instance_id']:r for r in bench}
    instances=[r['instance_id'] for r in bench]

    llm=list(csv.DictReader((root/'results/raw/mistral_small_2603_v1.4.csv').open(encoding='utf-8',newline='')))
    pred={}
    for s in ('S1','S2','S3'):
        pred[s]={r['instance_id']:parse_json_list(r['predicted_capabilities']) for r in llm if r['strategy_id']==s and r['status']=='DONE'}
    b0=list(csv.DictReader((root/'results/raw/baseline_rule_based_v1.0_predictions.csv').open(encoding='utf-8',newline='')))
    pred['B0']={r['instance_id']:split_semicolon(r['baseline_capabilities']) for r in b0}

    strategy=[]
    for s in ('B0','S1','S2','S3'):
        strategy.append({'method':s,**score(pred[s],gold,instances)})
    write_csv(root/'results/processed/strategy_metrics_v1.4.csv',strategy)

    reps=[]
    for rep in ('R1_NL','R2_STRUCTURED','R3_API_TELEMETRY'):
        ids=[iid for iid in instances if meta[iid]['representation']==rep]
        for s in ('B0','S1','S2','S3'):
            reps.append({'method':s,'representation':rep,**score(pred[s],gold,ids)})
    write_csv(root/'results/processed/by_representation_v1.4.csv',reps)

    fams=[]
    families=[]
    for r in bench:
        if r['device_family'] not in families: families.append(r['device_family'])
    for fam in families:
        ids=[iid for iid in instances if meta[iid]['device_family']==fam]
        for s in ('B0','S1','S2','S3'):
            fams.append({'method':s,'device_family':fam,**score(pred[s],gold,ids)})
    write_csv(root/'results/processed/by_family_v1.4.csv',fams)

    pc=[]
    for c in CAPS:
        row={'capability':c,'support':sum(c in gold[iid] for iid in instances)}
        for s in ('B0','S1','S2','S3'):
            tp=fp=fn=0
            for iid in instances:
                g=gold[iid]; p=pred[s][iid]
                tp+=int(c in g and c in p); fp+=int(c not in g and c in p); fn+=int(c in g and c not in p)
            row[f'{s}_f1']=f1(tp,fp,fn)
        pc.append(row)
    write_csv(root/'results/processed/per_capability_f1_v1.4.csv',pc)

    # Pairwise exact-match McNemar comparisons on the same 108 instances.
    exact={s:{iid:int(pred[s][iid]==gold[iid]) for iid in instances} for s in ('B0','S1','S2','S3')}
    pair=[]
    pairs=[('S1','S2'),('S1','S3'),('S2','S3'),('B0','S1'),('B0','S2'),('B0','S3')]
    for a,b in pairs:
        a_only=sum(exact[a][iid] and not exact[b][iid] for iid in instances)
        b_only=sum(exact[b][iid] and not exact[a][iid] for iid in instances)
        pair.append({'method_a':a,'method_b':b,'a_only_exact':a_only,'b_only_exact':b_only,'discordant_pairs':a_only+b_only,'exact_mcnemar_p':exact_mcnemar_p(a_only,b_only)})
    write_csv(root/'results/processed/pairwise_exact_match_mcnemar_v1.4.csv',pair)

    print('Generated processed CSV tables.')

if __name__=='__main__': main()
