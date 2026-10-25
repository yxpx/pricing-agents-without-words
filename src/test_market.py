"""Meaningful evaluator checks; runs entirely on CPU."""
import json,math,tempfile
from pathlib import Path
import numpy as np
from market import Market
from runner import World,make_record,run_study,ScriptedPolicy,scripted_controls,probabilities,scripted_action

def main():
 m=Market();b=m.benchmarks();assert b['nash_actions']==[3,3],b
 assert b['joint_actions']==[9,9],b
 for i in range(11):
  for j in range(11):
   o=m.outcome([i,j]);v1=math.exp((2-m.prices[i])/.25);v2=math.exp((2-m.prices[j])/.25)
   np.testing.assert_allclose(o['quantities'],[v1/(1+v1+v2),v2/(1+v1+v2)],rtol=1e-12)
   assert abs(sum(o['quantities'])+o['outside_share']-1)<1e-12
   assert o['consumer_surplus']>=0
 h=[{'round':0,'actions':[3,9],**m.outcome([3,9])}]
 own=make_record(m,h,0,'own_history')['state'];live=make_record(m,h,0,'live')['state']
 assert 'rival_price'not in own and 'rival_price'in live
 w=World('x','live',123,history=h);r=World.restore(w.snapshot())
 for i in range(2):np.testing.assert_array_equal(w.rng[i].random(10),r.rng[i].random(10))
 under=World('under','live',10,focal=0,history=[{'round':0,'actions':[4,8],**m.outcome([4,8])}])
 under.branch='one_round_undercut';under.branch_round=1
 assert scripted_action(under,1,1,m,b)==3
 under.history=[{'round':0,'actions':[0,8],**m.outcome([0,8])}]
 assert scripted_action(under,1,1,m,b)==0
 controls=scripted_controls(m,b,20)
 assert controls['trigger'][9]['actions']==[9,9] and controls['trigger'][11]['actions']==[3,3]
 with tempfile.TemporaryDirectory()as d:
  out=Path(d)/'study';manifest=run_study(ScriptedPolicy('nash'),out,markets=2,rounds=6,branch_rounds=2,donors=2,max_minutes=1)
  assert manifest['status']=='complete'
  records=[json.loads(s)for s in(out/'main.jsonl').read_text().splitlines()]
  assert len(records)==2*3*6*2
  assert all(r['action_index']==3 for r in records)
  assert sum(r['source']=='model'for r in records)==5*2*6
  parents=json.loads((out/'branch_parents.json').read_text());assert all(len(p['history'])==4 for p in parents)
  assert set(p['world_id']for p in parents)=={'live_000','live_001'}
  branches=[json.loads(s)for s in(out/'branches.jsonl').read_text().splitlines()]
  assert len(branches)==2*2*2*2
  assert all(r['round']in[4,5]for r in branches)
 print('PASS: all payoff cells, unique discrete benchmarks, observation isolation, RNG restoration, trigger control, and complete checkpointed study.')
if __name__=='__main__':main()
