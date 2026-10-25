"""Audit saved experiment evidence and compute numeric summaries, without model calls.
Usage: python experiments/analyze.py --run experiments/runs/<run>
"""
from __future__ import annotations
import argparse,hashlib,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from market import Market
from runner import make_record,write_json


def bootstrap(values,rng,reps):
    x=np.asarray(values,dtype=float)
    if not len(x):return {'n':0,'mean':None,'ci95':None}
    draws=x[rng.integers(len(x),size=(reps,len(x)))].mean(1)
    return {'n':len(x),'mean':float(x.mean()),'ci95':[float(z)for z in np.quantile(draws,[.025,.975])],
            'unit':'independent market (parent market for branch effects)'}


def hier_means(points,rng,reps,key='G'):
    groups=defaultdict(list)
    for p in points:groups[p['donor_id']].append(float(p[key]))
    ids=sorted(groups);out=[]
    for _ in range(reps):
        chosen=rng.choice(len(ids),size=len(ids),replace=True);vals=[]
        for i in chosen:
            x=groups[ids[int(i)]];vals.extend(rng.choice(x,size=len(x),replace=True).tolist())
        out.append(float(np.mean(vals)))
    return np.asarray(out),groups


def analyze(run,reps=10000):
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text());T=int(manifest['rounds']);N=int(manifest['markets_per_condition'])
    H=int(manifest['branch_rounds']);D=int(manifest['donors']);B=T-H;window=int(manifest['history_window'])
    errors=[];warnings=[];stage_counts={};groups={};sources=Counter();hashes={}
    market=Market(prices=tuple(manifest['market']['grid']));bench=market.benchmarks()
    def check(ok,msg):
        if not ok:errors.append(msg)
    check(np.allclose(bench['nash']['profits'],manifest['market']['nash']['profits']), 'Manifest Nash payoff disagrees with current exact market')
    check(np.allclose(bench['joint']['profits'],manifest['market']['joint']['profits']), 'Manifest joint payoff disagrees with current exact market')
    designs={'donors':(D,T,0),'main':(3*N,T,0),'branches':(2*N,H,B)}
    for stage,(nw,nr,start)in designs.items():
        path=run/f'{stage}.jsonl';rows=[];unique=set();by=defaultdict(dict)
        if path.exists():
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line_no,line in enumerate(path.read_text().splitlines(),1):
                if not line.strip():continue
                try:r=json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f'{stage}:{line_no}: malformed/partial JSON');continue
                key=(r['market_id'],r['round'],r['firm'])
                check(key not in unique,f'{stage}: duplicate decision key {key}');unique.add(key)
                check(r.get('stage')==stage,f'{stage}:{key}: mismatched recorded stage')
                check(r['firm']in[0,1]and r['focal']in[0,1],f'{stage}:{key}: invalid firm/focal')
                check(start<=r['round']<start+nr,f'{stage}:{key}: out-of-horizon round')
                check(0<=r['action_index']<11,f'{stage}:{key}: invalid action index')
                check(np.allclose(r['menu_prices'],market.prices),f'{stage}:{key}: menu-price mismatch')
                if r['source']=='model':
                    p=np.asarray(r['probabilities'],dtype=float)
                    check(p.shape==(11,)and np.isfinite(p).all()and(p>=0).all()and abs(float(p.sum())-1)<1e-5,f'{stage}:{key}: invalid probability vector')
                    if p.shape==(11,):check(p[r['action_index']]>0,f'{stage}:{key}: sampled zero-probability choice')
                    obs=r.get('observation')or{};options=obs.get('questions',{}).get('price',{}).get('criteria',{})
                    check(set(options)==set('ABCDEFGHIJK'),f'{stage}:{key}: invalid option IDs')
                    for label,description in options.items():
                        found=re.search(r'price\s+([\d.]+)',description)
                        check(bool(found)and abs(float(found.group(1).rstrip('.'))-market.prices[ord(label)-65])<1e-6,f'{stage}:{key}: option numeric mapping mismatch {label}')
                else:
                    check(r['source']=='scripted'and r.get('probabilities')is None,f'{stage}:{key}: invalid scripted record')
                rows.append(r);by[r['market_id']][(r['round'],r['firm'])]=r;sources[r['source']]+=1
        complete=[]
        for mid,entries in by.items():
            wanted={(rr,ff)for rr in range(start,start+nr)for ff in range(2)}
            if set(entries)==wanted:complete.append(mid)
            for rr in sorted(set(k[0]for k in entries)):
                pair=[entries.get((rr,j))for j in range(2)]
                if not all(pair):continue
                actions=[r['action_index']for r in pair];o=market.outcome(actions)
                for r in pair:
                    for field in ['prices','quantities','profits','consumer_surplus','outside_share']:
                        check(np.allclose(r[field],o[field],rtol=1e-9,atol=1e-10),f'{stage}:{mid}:{rr}:{r["firm"]}: {field} fails exact recomputation')
                check(pair[0]['focal']==pair[1]['focal'],f'{stage}:{mid}:{rr}: inconsistent focal role')
        stage_counts[stage]={'actual_records':len(rows),'expected_records':nw*nr*2,'unique_decisions':len(unique),
                             'actual_markets':len(by),'expected_markets':nw,'complete_markets':len(complete),
                             'complete_market_ids':sorted(complete),'missing_records':max(0,nw*nr*2-len(unique))}
        groups[stage]=by
        if stage in manifest.get('stats',{}):
            actual_calls=sum(r['source']=='model'for r in rows)
            check(actual_calls==manifest['stats'][stage]['model_decisions'],f'{stage}: manifest model-decision count mismatch')
        if len(complete)!=nw or len(rows)!=nw*nr*2:warnings.append(f'{stage} incomplete: {len(complete)}/{nw} complete markets')
    parents={p['world_id']:p for p in json.loads((run/'branch_parents.json').read_text())}if(run/'branch_parents.json').exists()else{}
    donor_paths={p['id']:p['actions']for p in json.loads((run/'donor_paths.json').read_text())}if(run/'donor_paths.json').exists()else{}
    for did,path in donor_paths.items():
        check(len(path)==T,f'{did}: donor path horizon mismatch')
    expected_names={f'{c}_{i:03}'for i in range(N)for c in ['live','own_history','replay']}
    check(set(groups['main']).issubset(expected_names),'Unexpected main market IDs')
    check(set(groups['donors']).issubset({f'donor_{i:03}'for i in range(D)}),'Unexpected donor IDs')
    expected_branches={f'live_{i:03}__{b}'for i in range(N)for b in ['one_round_undercut','competitive_rival']}
    check(set(groups['branches']).issubset(expected_branches),'Unexpected branch IDs')
    # Validate replay assignment and observations against only the available stage history.
    for stage,by in groups.items():
        for mid,entries in by.items():
            first=next(iter(entries.values()));condition=first['condition'];focal=first['focal'];history=[]
            if stage=='branches':
                parent_id=first['parent_id'];check(parent_id in parents,f'{mid}: missing branch parent snapshot')
                check(parent_id in groups['main'],f'{mid}: branch parent absent from main data')
                if parent_id in parents:
                    snap=parents[parent_id];history=snap['history'].copy()
                    check(len(history)==B,f'{mid}: wrong parent horizon')
                    for h in history:
                        for firm in range(2):
                            old=groups['main'].get(parent_id,{}).get((h['round'],firm))
                            if old is not None:check(abs(h['prices'][firm]-old['prices'][firm])<1e-10,f'{mid}: parent snapshot differs from raw main history')
            for rr in sorted(set(k[0]for k in entries)):
                pair=[entries.get((rr,j))for j in range(2)]
                if not all(pair):continue
                for firm,r in enumerate(pair):
                    expected_source='model'
                    if condition=='replay'and firm!=focal:expected_source='scripted'
                    if stage=='branches'and firm!=focal and (r['branch']=='competitive_rival'or(r['branch']=='one_round_undercut'and rr==B)):expected_source='scripted'
                    check(r['source']==expected_source,f'{stage}:{mid}:{rr}:{firm}: unexpected model/scripted source')
                    if r['source']=='model':
                        want=make_record(market,history,firm,condition,window)
                        check(r['observation']['state']==want['state'],f'{stage}:{mid}:{rr}:{firm}: observation history mismatch')
                    if condition=='replay'and firm!=focal:
                        did=r['donor_id'];check(did in donor_paths,f'{mid}: missing donor {did}')
                        if did in donor_paths:check(r['action_index']==donor_paths[did][rr],f'{mid}:{rr}: replay action differs from donor')
                    if stage=='branches'and firm!=focal:
                        if r['branch']=='competitive_rival':check(r['action_index']==bench['nash_actions'][firm],f'{mid}:{rr}: incorrect competitive script')
                        if r['branch']=='one_round_undercut'and rr==B and history:
                            action=max(0,min(history[-1]['actions'])-1)
                            check(r['action_index']==action,f'{mid}:{rr}: incorrect forced undercut')
                            check(r['undercut_noop']==(min(history[-1]['actions'])==0),f'{mid}:{rr}: incorrect floor no-op flag')
                history.append({'round':rr,'actions':[r['action_index']for r in pair],**market.outcome([r['action_index']for r in pair])})
    rng=np.random.default_rng(20261025);points=[]
    for mid in stage_counts['main']['complete_market_ids']:
        entries=groups['main'][mid];first=next(iter(entries.values()));f=first['focal'];term=[entries[(rr,f)]for rr in range(T-H,T)]
        pn=bench['nash']['profits'][f];pj=bench['joint']['profits'][f]
        pi=float(np.mean([r['profits'][f]for r in term]));pr=float(np.mean([r['prices'][f]for r in term]));cs=float(np.mean([r['consumer_surplus']for r in term]))
        ep=float(np.mean([np.dot(r['probabilities'],market.prices)for r in term]))
        points.append({'market_id':mid,'condition':first['condition'],'focal':f,'seed':first['seed'],'donor_id':first['donor_id'],
                       'terminal_rounds':H,'G':(pi-pn)/(pj-pn),'profit':pi,'price':pr,'expected_price':ep,'consumer_surplus':cs})
    conditions={}
    for c in ['live','own_history','replay']:
        pp=[p for p in points if p['condition']==c]
        conditions[c]={key:bootstrap([p[key]for p in pp],rng,reps)for key in ['G','profit','price','expected_price','consumer_surplus']}
    live=[p for p in points if p['condition']=='live'];replay=[p for p in points if p['condition']=='replay'];contrast=None;sensitivity={}
    if live and replay:
        lx=np.array([p['G']for p in live]);rx=np.array([p['G']for p in replay])
        ld=lx[rng.integers(len(lx),size=(reps,len(lx)))].mean(1);rd=rx[rng.integers(len(rx),size=(reps,len(rx)))].mean(1)
        hd,donor_groups=hier_means(replay,rng,reps)
        contrast={'estimate':float(lx.mean()-rx.mean()),'live_n':len(lx),'replay_n':len(rx),
                  'iid_market_ci95':[float(z)for z in np.quantile(ld-rd,[.025,.975])],
                  'replay_hierarchical_donor_ci95':[float(z)for z in np.quantile(ld-hd,[.025,.975])],
                  'independent_replay_donors':len(donor_groups),'small_donor_warning':len(donor_groups)<4}
        sensitivity={'donor_reuse_counts':{k:len(v)for k,v in donor_groups.items()},
                     'leave_one_donor_out_live_minus_replay_G':{k:float(lx.mean()-np.mean([p['G']for p in replay if p['donor_id']!=k]))for k in donor_groups if any(p['donor_id']!=k for p in replay)}}
    effects=[]
    for mid in stage_counts['branches']['complete_market_ids']:
        e=groups['branches'][mid];first=next(iter(e.values()));pid=first['parent_id'];f=first['focal']
        if pid not in stage_counts['main']['complete_market_ids']:continue
        control=groups['main'][pid]
        for name,length in [('first5',min(5,H)),('all',H)]:
            rr=range(B,B+length);diffs={}
            for metric in ['expected_price','price','profit','consumer_surplus']:
                def value(r):
                    if metric=='expected_price':return float(np.dot(r['probabilities'],market.prices))
                    if metric=='price':return r['prices'][f]
                    if metric=='profit':return r['profits'][f]
                    return r['consumer_surplus']
                diffs[metric]=float(np.mean([value(e[(r,f)])-value(control[(r,f)])for r in rr]))
            effects.append({'parent_id':pid,'branch':first['branch'],'window':name,'rounds':length,'difference':diffs,
                            'undercut_noop':any(r['undercut_noop']for r in e.values())})
    branch_summaries={}
    for branch in ['one_round_undercut','competitive_rival']:
        branch_summaries[branch]={}
        for w in ['first5','all']:
            es=[e for e in effects if e['branch']==branch and e['window']==w]
            branch_summaries[branch][w]={m:bootstrap([e['difference'][m]for e in es],rng,reps)for m in ['expected_price','price','profit','consumer_surplus']}
    data_complete=all(s['complete_markets']==s['expected_markets']and s['actual_records']==s['expected_records']for s in stage_counts.values())
    validation={'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'run':str(run),'manifest_status':manifest.get('status'),
                'integrity_pass':not errors,'raw_data_complete':data_complete,'complete':not errors and data_complete and manifest.get('status')=='complete',
                'errors':errors,'warnings':warnings,'stages':stage_counts,'record_sources':dict(sources),'raw_sha256':hashes,
                'limitations':['Numeric menu consistency can be audited; model-internal logit-to-label mapping also requires adapter verification.',
                               'Replay donor reuse limits independence; hierarchical intervals are exploratory with few donors.']}
    summary={'run':str(run),'model_metadata':manifest.get('model_metadata'),'manifest_status':manifest.get('status'),'validation_complete':validation['complete'],
             'bootstrap_replicates':reps,'terminal_window':H,'benchmarks':bench,'completed_market_points':points,'conditions':conditions,
             'primary_live_minus_replay_G':contrast,'donor_sensitivity':sensitivity,'paired_branch_points':effects,'paired_branch_summaries':branch_summaries,
             'interpretation':'Observed pricing and contingent responses; no conclusion of legal agreement, intent, or collusion from similarity alone.'}
    write_json(run/'validation.json',validation);write_json(run/'summary.json',summary)
    return validation,summary


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--bootstrap',type=int,default=10000);args=a.parse_args()
    v,s=analyze(args.run,args.bootstrap)
    print(json.dumps({'complete':v['complete'],'integrity_pass':v['integrity_pass'],'errors':len(v['errors']),
                      'stages':v['stages'],'primary_live_minus_replay_G':s['primary_live_minus_replay_G']},indent=2))
    if not v['integrity_pass']:raise SystemExit(2)
if __name__=='__main__':main()
