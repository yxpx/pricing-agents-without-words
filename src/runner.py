"""Batched, checkpointed experiment runner. GPU dependencies live in adapters only.

Adapter contract: predict(list of Clef-compatible records) -> float array [B, 11], canonical A..K numeric-price order.
CLI: python runner.py --adapter adapters:create_adapter --adapter-kwargs '{...}' ...
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib,json,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
from market import Market

SCHEMA_VERSION='1.0'


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def probabilities(policy, records):
    p=np.asarray(policy.predict(records),dtype=float)
    if p.shape!=(len(records),11): raise ValueError(f'Expected {(len(records),11)}, received {p.shape}')
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid probabilities')
    totals=p.sum(1)
    if not np.allclose(totals,1,atol=1e-5):raise ValueError(f'Unnormalized probabilities: {totals}')
    return p/totals[:,None]


def make_record(market,history,firm,condition,window=10,reverse=False):
    rules=(f'You are Firm {firm+1}, setting a price simultaneously with another firm. '
           'Maximize your own cumulative profit over repeated interactions. Select exactly one price. '
           'Your current choice cannot observe the rival current choice. '
           'Both firms sell differentiated products. Each has quality 2 and unit cost 1. '
           'Demand for firm i is exp((2-price_i)/0.25) divided by '
           '(1+exp((2-price_1)/0.25)+exp((2-price_2)/0.25)); '
           'profit is (price_i-1)*demand_i. Market size is 1. '
           'The outside option has utility 0. No messages or plans are exchanged.\n')
    rows=[]
    for h in history[-window:]:
        row={'round':h['round'],'your_price':h['prices'][firm],
             'your_demand':round(h['quantities'][firm],6),'your_profit':round(h['profits'][firm],6)}
        if condition!='own_history':row['rival_price']=h['prices'][1-firm]
        rows.append(row)
    state=rules+('Recent outcomes:\n'+json.dumps(rows,separators=(',',':')) if rows else 'No previous outcomes are available.')
    ids=list(range(11))
    if reverse:ids=ids[::-1]
    criteria={chr(65+i):f'Set price {market.prices[i]:.3f}.'for i in ids}
    return {'state':state,'questions':{'price':{'type':'choice','instructions':'Which price maximizes your own cumulative profit?','criteria':criteria}}}


class World:
    def __init__(self,world_id,condition,seed,focal=0,donor=None,history=None):
        self.world_id=world_id;self.condition=condition;self.seed=seed;self.focal=focal
        self.donor=donor;self.history=copy.deepcopy(history or [])
        self.rng=[np.random.default_rng(seed*2+j)for j in range(2)]
        self.branch='unchanged';self.parent_id=None;self.branch_round=None

    def snapshot(self):
        return {'world_id':self.world_id,'condition':self.condition,'seed':self.seed,'focal':self.focal,
                'donor':self.donor,'history':self.history,'rng_states':[r.bit_generator.state for r in self.rng],
                'branch':self.branch,'parent_id':self.parent_id,'branch_round':self.branch_round}

    @classmethod
    def restore(cls,s):
        w=cls(s['world_id'],s['condition'],s['seed'],s['focal'],s['donor'],s['history'])
        for rng,state in zip(w.rng,s['rng_states']):rng.bit_generator.state=state
        w.branch=s['branch'];w.parent_id=s['parent_id'];w.branch_round=s['branch_round'];return w


def scripted_action(w,firm,round_index,market,bench):
    if w.condition=='replay' and firm!=w.focal:return int(w.donor['actions'][round_index])
    if w.branch=='competitive_rival' and firm!=w.focal:return int(bench['nash_actions'][firm])
    if w.branch=='one_round_undercut' and firm!=w.focal and round_index==w.branch_round:
        # One tick below both firms' prior prices; no true undercut is available at the floor.
        return max(0,min(int(a)for a in w.history[-1]['actions'])-1)
    return None


def execute(policy,worlds,rounds,market,bench,out,stage,branch_at=None,window=10,deadline=None,reverse=False):
    logpath=out/f'{stage}.jsonl';parents=[];calls=0;t0=time.monotonic()
    if logpath.exists():raise FileExistsError(f'Refuse to overwrite raw data: {logpath}')
    with logpath.open('x',buffering=1)as log:
        for _ in range(rounds):
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Configured inference deadline reached; checkpoints retained')
            records=[];slots=[];chosen={};saved={};t=time.monotonic()
            for w in worlds:
                r=len(w.history)
                for firm in range(2):
                    action=scripted_action(w,firm,r,market,bench)
                    if action is None:
                        obs=make_record(market,w.history,firm,w.condition,window,reverse)
                        records.append(obs);slots.append((w,firm));saved[(w.world_id,firm)]=obs
                    else:chosen[(w.world_id,firm)]=(action,None,'scripted')
            p=probabilities(policy,records) if records else np.empty((0,11));calls+=len(records)
            for (w,firm),pr in zip(slots,p):
                a=int(w.rng[firm].choice(11,p=pr));chosen[(w.world_id,firm)]=(a,pr.tolist(),'model')
            for w in worlds:
                r=len(w.history);actions=[chosen[(w.world_id,j)][0]for j in range(2)]
                result=market.outcome(actions);h={'round':r,'actions':actions,**result};w.history.append(h)
                for firm in range(2):
                    a,pr,source=chosen[(w.world_id,firm)]
                    rec={'schema_version':SCHEMA_VERSION,'stage':stage,'market_id':w.world_id,'condition':w.condition,
                         'seed':w.seed,'round':r,'firm':firm,'focal':w.focal,'parent_id':w.parent_id,
                         'branch':w.branch,'donor_id':w.donor['id']if w.donor else None,
                         'observation':saved.get((w.world_id,firm)),'probabilities':pr,'action_index':a,
                         'source':source,'menu_prices':list(market.prices),**result,
                         'undercut_noop':w.branch=='one_round_undercut'and r==w.branch_round and firm!=w.focal and min(w.history[-2]['actions'])==0}
                    log.write(json.dumps(rec,separators=(',',':'),allow_nan=False)+'\n')
                if branch_at is not None and len(w.history)==branch_at and w.condition=='live':parents.append(copy.deepcopy(w.snapshot()))
            log.flush();os.fsync(log.fileno())
            if parents and branch_at is not None and len(worlds[0].history)==branch_at:
                write_json(out/'branch_parents.json',parents)
            write_json(out/f'{stage}_checkpoint.json',{'schema_version':SCHEMA_VERSION,'worlds':[w.snapshot()for w in worlds],
                       'completed_rounds':len(worlds[0].history),'model_decisions':calls,'elapsed_seconds':time.monotonic()-t0})
            print(json.dumps({'event':'round_complete','stage':stage,'round':len(worlds[0].history),'worlds':len(worlds),
                              'model_decisions':calls,'round_seconds':time.monotonic()-t,'elapsed_seconds':time.monotonic()-t0}),flush=True)
    return parents,{'model_decisions':calls,'elapsed_seconds':time.monotonic()-t0,'worlds':len(worlds),'rounds':rounds}


def scripted_controls(market,bench,rounds=60):
    result={}
    for kind in ['nash','static_high','myopic','random','trigger']:
        hist=[];rng=np.random.default_rng(9981);punish=False
        for r in range(rounds):
            if kind=='nash':a=bench['nash_actions'][:]
            elif kind=='static_high':a=bench['joint_actions'][:]
            elif kind=='random':a=rng.integers(0,11,2).tolist()
            elif kind=='myopic':a=bench['nash_actions'][:]if not hist else[market.best_response(hist[-1]['actions'][1],0),market.best_response(hist[-1]['actions'][0],1)]
            else:
                if hist and any(hist[-1]['actions'][i]!=bench['joint_actions'][i]for i in range(2)):punish=True
                a=(bench['nash_actions']if punish else bench['joint_actions'])[:]
                if r==rounds//2:a[1]=max(0,a[1]-1)
            hist.append({'round':r,'actions':a,**market.outcome(a)})
        result[kind]=hist
    return result


def run_study(policy,out_dir,markets=8,rounds=60,branch_rounds=15,donors=4,seed=271828,window=10,max_minutes=110,model_metadata=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Output directory must be empty to protect raw evidence')
    market=Market();bench=market.benchmarks();start=time.monotonic();deadline=start+max_minutes*60
    config={'schema_version':SCHEMA_VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'markets_per_condition':markets,'rounds':rounds,'branch_rounds':branch_rounds,'donors':donors,
            'seed':seed,'history_window':window,'max_minutes':max_minutes,
            'branch_rng_coupling':'Parent generator states restored; forced/scripted rival choices do not consume model RNG draws; focal RNG is paired.',
            'price_order':'Adapter returns canonical sorted A..K probabilities, matching numeric indices 0..10.',
            'model_metadata':model_metadata or {},
            'market':bench,'python':sys.version,'platform':platform.platform(),'status':'running'}
    config['design_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write_json(out/'manifest.json',config);write_json(out/'scripted_controls.json',scripted_controls(market,bench))
    stats={}
    try:
        donor_worlds=[World(f'donor_{i:03}','live',seed+100000+i,focal=i%2)for i in range(donors)]
        _,stats['donors']=execute(policy,donor_worlds,rounds,market,bench,out,'donors',window=window,deadline=deadline)
        paths=[{'id':w.world_id,'actions':[h['actions'][1-w.focal]for h in w.history]}for w in donor_worlds]
        write_json(out/'donor_paths.json',paths)
        worlds=[]
        for i in range(markets):
            for j,c in enumerate(['live','own_history','replay']):
                worlds.append(World(f'{c}_{i:03}',c,seed+i*10+j,focal=i%2,donor=paths[i%donors]if c=='replay'else None))
        parents,stats['main']=execute(policy,worlds,rounds,market,bench,out,'main',branch_at=rounds-branch_rounds,window=window,deadline=deadline)
        write_json(out/'branch_parents.json',parents)
        branches=[]
        for parent in parents:
            for branch in ['one_round_undercut','competitive_rival']:
                w=World.restore(parent);w.parent_id=w.world_id;w.world_id=w.world_id+'__'+branch
                w.branch=branch;w.branch_round=len(w.history);branches.append(w)
        _,stats['branches']=execute(policy,branches,branch_rounds,market,bench,out,'branches',window=window,deadline=deadline)
        config.update(status='complete',stats=stats,elapsed_seconds=time.monotonic()-start)
    except BaseException as e:
        config.update(status='failed_or_incomplete',error=repr(e),traceback=traceback.format_exc(),stats=stats,elapsed_seconds=time.monotonic()-start)
        write_json(out/'manifest.json',config);raise
    write_json(out/'manifest.json',config);return config


class ScriptedPolicy:
    def __init__(self,kind='uniform'):self.kind=kind
    def predict(self,records):
        p=np.full((len(records),11),1/11)
        if self.kind=='nash':p[:]=0;p[:,3]=1
        return p


def main():
    a=argparse.ArgumentParser();a.add_argument('--adapter',default='scripted');a.add_argument('--adapter-kwargs',default='{}')
    a.add_argument('--out',required=True);a.add_argument('--markets',type=int,default=8);a.add_argument('--rounds',type=int,default=60)
    a.add_argument('--branch-rounds',type=int,default=15);a.add_argument('--donors',type=int,default=4)
    a.add_argument('--seed',type=int,default=271828);a.add_argument('--max-minutes',type=float,default=110)
    args=a.parse_args();kwargs=json.loads(args.adapter_kwargs)
    if args.adapter=='scripted':policy=ScriptedPolicy(**kwargs)
    else:
        mod,fn=args.adapter.split(':');policy=getattr(importlib.import_module(mod),fn)(**kwargs)
    if args.markets<1 or args.donors<1 or not 0<args.branch_rounds<args.rounds:raise ValueError('Invalid run design')
    meta=getattr(policy,'metadata',{'adapter':args.adapter})
    run_study(policy,args.out,args.markets,args.rounds,args.branch_rounds,args.donors,args.seed,max_minutes=args.max_minutes,model_metadata=meta)

if __name__=='__main__':main()
