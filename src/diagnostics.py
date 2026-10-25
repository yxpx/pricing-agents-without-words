"""Small fixed-state competency/menu audit; separate from market outcomes."""
import json,time
from pathlib import Path
import numpy as np
from market import Market
from runner import make_record,probabilities


def run_diagnostics(policy,out):
    m=Market();rows=[];records=[]
    # Exactly solvable one-step problems, not a model-judged competency label.
    for firm in range(2):
        for rival in range(11):
            r=make_record(m,[],firm,'live')
            r['state']=(f'This is a ONE-SHOT pricing decision, with no future rounds. You are firm {firm+1}. '
                f'The rival has irrevocably committed to price {m.prices[rival]:.3f}. '
                'Both products have quality 2 and unit cost 1. Demand_i=exp((2-price_i)/0.25) / '
                '(1+exp((2-price_1)/0.25)+exp((2-price_2)/0.25)). Profit_i=(price_i-1)*demand_i. '
                'Select the price maximizing your own immediate profit.')
            r['questions']['price']['instructions']='Which price maximizes immediate profit against the committed rival price?'
            records.append(r);rows.append({'type':'one_shot','firm':firm,'rival_index':rival})
    for firm in range(2):
        for name,history in [('empty',[]),('competitive',[{'round':i,'actions':[3,3],**m.outcome([3,3])}for i in range(10)]),('high',[{'round':i,'actions':[9,9],**m.outcome([9,9])}for i in range(10)])]:
            for rev in [False,True]:
                records.append(make_record(m,history,firm,'live',reverse=rev));rows.append({'type':'menu_order','firm':firm,'history':name,'reverse':rev})
    ps=probabilities(policy,records)
    result=[]
    for row,r,p in zip(rows,records,ps):
        row={**row,'observation':r,'probabilities':p.tolist(),'expected_price':float(p@np.array(m.prices)),'argmax_price_index':int(p.argmax())}
        if row['type']=='one_shot':
            f=row['firm'];v=row['rival_index'];pay=np.array([m.outcome([a,v] if f==0 else [v,a])['profits'][f]for a in range(11)])
            row.update(best_action=int(pay.argmax()),argmax_correct=bool(p.argmax()==pay.argmax()),expected_profit_regret=float(pay.max()-p@pay))
        result.append(row)
    out=Path(out);out.write_text(json.dumps({'created':time.time(),'model':policy.metadata,'records':result},indent=2))
    return {'one_shot_cases':22,'argmax_accuracy':float(np.mean([r['argmax_correct']for r in result if r['type']=='one_shot'])),'path':str(out)}
