"""Secondary common-horizon comparison; no architecture-only causal claim."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from market import Market
from runner import write_json

def points(run):
    v=json.loads((run/'validation.json').read_text());assert v['complete'],f'Incomplete or unaudited run {run}'
    m=Market();b=m.benchmarks();groups=defaultdict(list)
    for line in (run/'main.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['firm']==r['focal'] and 45<=r['round']<60:groups[r['market_id']].append(r)
    out=[]
    for mid,rows in groups.items():
        assert len(rows)==15 and {r['round']for r in rows}==set(range(45,60))
        f=rows[0]['focal'];pi=float(np.mean([r['profits'][f]for r in rows]));pn=b['nash']['profits'][f];pj=b['joint']['profits'][f]
        out.append({'market_id':mid,'condition':rows[0]['condition'],'donor_id':rows[0]['donor_id'],'G':(pi-pn)/(pj-pn),'price':float(np.mean([r['prices'][f]for r in rows]))})
    return out

def draws(rows,rng,reps):
    if rows[0]['condition']!='replay':
        x=np.array([r['G']for r in rows]);return x[rng.integers(len(x),size=(reps,len(x)))].mean(1)
    groups=defaultdict(list)
    for r in rows:groups[r['donor_id']].append(r['G'])
    keys=list(groups);result=[]
    for _ in range(reps):
        selected=rng.choice(len(keys),size=len(keys),replace=True);values=[]
        for i in selected:
            x=groups[keys[i]];values.extend(rng.choice(x,size=len(x),replace=True))
        result.append(np.mean(values))
    return np.array(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clef',required=True);p.add_argument('--qwen',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    c=points(Path(a.clef));q=points(Path(a.qwen));rng=np.random.default_rng(20261025);contrasts={}
    for condition in ['live','own_history','replay']:
        x={r['market_id']:r for r in c if r['condition']==condition};y={r['market_id']:r for r in q if r['condition']==condition}
        ids=sorted(set(x)&set(y));assert ids and all(x[i]['market_id']==y[i]['market_id']for i in ids)
        differences=np.array([y[i]['G']-x[i]['G']for i in ids])
        interval=None
        if condition!='replay':
            draws_=differences[rng.integers(len(ids),size=(10000,len(ids)))].mean(1)
            interval=np.quantile(draws_,[.025,.975]).tolist()
        contrasts[condition]={'clef_available_n':len(x),'qwen_available_n':len(y),'matched_seed_pairs':len(ids),'qwen_minus_clef_G':float(differences.mean()),'paired_ci95':interval,'uncertainty_note':'Shared RNG seeds are paired. Replay CI omitted: small crossed donor pools create dependence beyond seed pairs.' if condition=='replay'else 'Bootstrap resamples complete matched seed pairs.'}
    result={'status':'secondary_exploratory','window_zero_based':[45,59],'horizon':60,'contrasts':contrasts,'clef_points':c,'qwen_points':q,'limitations':['Training and readouts differ; this is not an architecture-only intervention.','Clef uses BF16 and Qwen FP32 after numerical stability checks.','Common random seeds require paired cross-model analysis.','Replay donor pools are small and crossed; no cross-model replay confidence interval is claimed.','Original primary analyses retain their different frozen terminal windows.']}
    write_json(a.out,result);print(json.dumps(contrasts,indent=2))
