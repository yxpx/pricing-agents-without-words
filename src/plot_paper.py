"""Reproducible paper figures from completed, locally audited runs. No inference.

Run from this repository: python src/plot_paper.py
PDFs go to outputs/figures; raster previews go to outputs/previews.
Pointwise uncertainty is not a simultaneous confidence band. Fixed-state probes
are deterministic cases and receive no inferential error bars.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.ticker import MultipleLocator
import numpy as np
from analyze import hier_means
from market import Market

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'; OUT.mkdir(parents=True,exist_ok=True)
PREVIEW=ROOT/'outputs/previews'; PREVIEW.mkdir(parents=True,exist_ok=True)
REPS=10000
COLORS={'live':'#245C85','own_history':'#BB6A35','replay':'#6E777D',
        'Clef-flash':'#245C85','Qwen3.5-9B':'#BB6A35',
        'empty':'#777777','competitive':'#245C85','high':'#BB6A35',
        'one_round_undercut':'#BB6A35','competitive_rival':'#245C85'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral'],
 'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'lines.linewidth':1.35,'xtick.major.width':.55,'ytick.major.width':.55,
 'grid.linewidth':.4,'grid.alpha':.2,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.transparent':False,'figure.facecolor':'white','axes.facecolor':'white'})
DATA={}; EVIDENCE={'bootstrap_reps':REPS,'seed':20261026,'sources':{},'models':{}}


def read(path):
 return json.loads(path.read_text())


def rows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def boot(x,seed=20261026):
 x=np.asarray(x);rng=np.random.default_rng(seed)
 draws=np.mean(x[rng.integers(len(x),size=(REPS,len(x)))],axis=1)
 return np.mean(x,axis=0),np.quantile(draws,[.025,.975],axis=0)


def clean(ax):
 ax.grid(axis='y',zorder=0);ax.tick_params(length=3,pad=3)


def save(fig,name):
 fig.savefig(OUT/f'{name}.pdf',bbox_inches=None)
 fig.savefig(PREVIEW/f'{name}.png',dpi=180)
 plt.close(fig)


for name,rel in [('Clef-flash','data/clef'),('Qwen3.5-9B','data/qwen')]:
 path=ROOT/rel;v=read(path/'final/validation.json');s=read(path/'final/summary.json');m=read(path/'final/manifest.json')
 assert v['complete'] and v['integrity_pass'] and s['validation_complete']
 for file,h in v['raw_sha256'].items():
  assert hashlib.sha256((path/'final'/file).read_bytes()).hexdigest()==h
 EVIDENCE['sources'][name]={'run':rel+'/final','raw_sha256':v['raw_sha256']}
 raw=rows(path/'final/main.jsonl');br=rows(path/'final/branches.jsonl');diag=read(path/'diagnostics.json')['records']
 main={(r['market_id'],r['round']):r for r in raw if r['firm']==r['focal']}
 branch=defaultdict(dict)
 for r in br:
  if r['firm']==r['focal']:branch[r['market_id']][r['round']]=r
 prices=np.array(m['market']['grid']);T=m['rounds'];H=m['branch_rounds'];N=m['markets_per_condition']
 curves={}
 for cond in ['live','own_history']:
  x=np.array([[np.dot(main[(f'{cond}_{i:03}',t)]['probabilities'],prices)for t in range(T)]for i in range(N)])
  curves[cond]=x
 DATA[name]={'summary':s,'manifest':m,'raw':main,'branches':branch,'diag':diag,'prices':prices,'curves':curves,'path':path}
 EVIDENCE['models'][name]={'terminal_summary':s['conditions'],'primary_contrast':s['primary_live_minus_replay_G']}

# Figure 1: trajectory and terminal profit; same scales within rows.
fig,axes=plt.subplots(2,2,figsize=(5.5,4.45),sharey='row')
fig.subplots_adjust(left=.105,right=.985,bottom=.105,top=.875,hspace=.55,wspace=.19)
for col,(name,d) in enumerate(DATA.items()):
 ax=axes[0,col];T=d['manifest']['rounds'];H=d['manifest']['branch_rounds']
 ax.axvspan(T-H+1,T,color='#E8EAEC',alpha=.65,zorder=0)
 for cond in ['live','own_history']:
  mean,ci=boot(d['curves'][cond]);t=np.arange(1,T+1)
  ax.fill_between(t,*ci,color=COLORS[cond],alpha=.13,linewidth=0)
  ax.plot(t,mean,color=COLORS[cond],label=cond)
 ax.axhline(1.475,color='#454545',ls='--',lw=.8)
 ax.axhline(1.925,color='#454545',ls=':',lw=.8)
 ax.set(xlim=(1,T),ylim=(1.35,1.98),xlabel='Round',title=f"{'a' if col==0 else 'b'}  {name} ($n={d['manifest']['markets_per_condition']}$)")
 ax.set_yticks([1.4,1.475,1.7,1.925]);ax.set_yticklabels(['1.40','1.475','1.70','1.925'])
 clean(ax)
 ax=axes[1,col]
 ax.axhline(0,color='#454545',ls='--',lw=.8)
 ax.axhline(1,color='#454545',ls=':',lw=.8)
 replay_cis={}
 for i,cond in enumerate(['live','own_history','replay']):
  pp=[p for p in d['summary']['completed_market_points']if p['condition']==cond]
  vals=np.array([p['G']for p in pp]);jitter=np.random.default_rng(29+i).uniform(-.18,.18,len(vals))
  ax.scatter(i+jitter,vals,s=14,color=COLORS[cond],alpha=.52,lw=0,zorder=3)
  mean=float(vals.mean())
  if cond=='replay':
   hd,_=hier_means(pp,np.random.default_rng(20261026),REPS);ci=np.quantile(hd,[.025,.975]);replay_cis['mean']=mean;replay_cis['ci95']=ci.tolist()
  else:ci=np.array(d['summary']['conditions'][cond]['G']['ci95'])
  ax.errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D',ms=4.8,color='#202020',capsize=3,elinewidth=1.1,zorder=5)
 EVIDENCE['models'][name]['replay_hierarchical_G']=replay_cis
 ax.set(xlim=(-.5,2.5),ylim=(-.75,1.08),xticks=[0,1,2],xticklabels=['Live','Own history','Recorded'],title=f"{'c' if col==0 else 'd'}  Terminal {H} rounds")
 clean(ax)
axes[0,0].set_ylabel('Expected price')
axes[1,0].set_ylabel('Profit gain $G$')
handles=[Line2D([],[],color=COLORS['live'],label='Live'),Line2D([],[],color=COLORS['own_history'],label='Own history'),Line2D([],[],color='#454545',ls='--',lw=.8,label='Competitive'),Line2D([],[],color='#454545',ls=':',lw=.8,label='Joint profit')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.995),ncol=4,frameon=False,columnspacing=1.2,handlelength=1.8)
save(fig,'outcomes')

# Added figure: full price probabilities over live rounds, preserving all original plots.
fig,axes=plt.subplots(1,2,figsize=(5.5,2.65),sharey=True)
fig.subplots_adjust(left=.115,right=.98,bottom=.32,top=.81,wspace=.17)
heat_matrices={}
for name,d in DATA.items():
 T=d['manifest']['rounds'];N=d['manifest']['markets_per_condition']
 cube=np.array([[d['raw'][(f'live_{i:03}',t)]['probabilities']for t in range(T)]for i in range(N)])
 assert cube.shape==(N,T,11)
 np.testing.assert_allclose(cube.sum(axis=2),1,atol=1e-6)
 heat_matrices[name]=cube.mean(axis=0)
# Same linear scale in both panels; no smoothing or per-panel normalization.
vmax=max(float(x.max())for x in heat_matrices.values())
for j,(name,d) in enumerate(DATA.items()):
 ax=axes[j];T=d['manifest']['rounds'];H=d['manifest']['branch_rounds'];pr=heat_matrices[name]
 edges=np.r_[d['prices']-.0375,d['prices'][-1]+.0375]
 im=ax.pcolormesh(np.arange(T+1)+.5,edges,pr.T,shading='flat',
                  cmap='Blues',vmin=0,vmax=vmax,edgecolors='none',antialiased=False)
 ax.set_xlim(.5,T+.5);ax.set_ylim(edges[0],edges[-1])
 ax.axhline(1.475,color='#606060',ls='--',lw=.8,
            path_effects=[patheffects.withStroke(linewidth=1.8,foreground='white')])
 ax.axhline(1.925,color='#606060',ls=':',lw=.8)
 ax.set(xlabel='Round',title=f"{'a'if j==0 else'b'}  {name} ($n={d['manifest']['markets_per_condition']}$)")
 ax.set_yticks([1.25,1.475,1.7,1.925,2.0]);ax.set_yticklabels(['1.250','1.475','1.700','1.925','2.000'])
 ax.tick_params(length=3,pad=3)
 stats={}
 for label,x in [('first_five',pr[:5]),('final_quarter',pr[-H:])]:
  stats[label]={'rounds':len(x),'mean_probabilities':x.mean(axis=0).tolist(),
                'competitive_band_1.400_to_1.550':float(x[:,2:5].sum(axis=1).mean()),
                'price_at_least_1.850':float(x[:,8:].sum(axis=1).mean())}
 EVIDENCE['models'][name]['live_probability_heatmap']={
  'mean_probabilities_by_round':pr.tolist(),'markets':d['manifest']['markets_per_condition'],
  'rounds':T,'price_grid':d['prices'].tolist(),'shared_linear_vmin':0,'shared_linear_vmax':vmax,
  'windows':stats}
axes[0].set_ylabel('Available price')
cax=fig.add_axes([.28,.105,.53,.045])
cbar=fig.colorbar(im,cax=cax,orientation='horizontal')
cbar.set_label('Mean action probability',labelpad=2)
cbar.ax.tick_params(labelsize=7.5,length=2,pad=2)
handles=[Line2D([],[],color='#606060',ls='--',lw=.8,label='Competitive price (1.475)'),
         Line2D([],[],color='#606060',ls=':',lw=.8,label='Joint-profit price (1.925)')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1.015),ncol=2,
           frameon=False,columnspacing=1.4,handlelength=2)
save(fig,'price_heatmap')

# Figure 2: fixed-history distributions. No confidence interval from deterministic cases.
fig,axes=plt.subplots(1,2,figsize=(5.5,2.15),sharey=True)
fig.subplots_adjust(left=.1,right=.985,bottom=.22,top=.73,wspace=.17)
for j,(name,d) in enumerate(DATA.items()):
 ax=axes[j];stats={}
 for h,label,ls in [('empty','No history',':'),('competitive','Competitive history','-'),('high','High-price history','-')]:
  pp=[r for r in d['diag']if r['type']=='menu_order'and r['history']==h and not r['reverse']]
  assert len(pp)==2
  probs=np.array([r['probabilities']for r in pp]);mean=probs.mean(0)
  ax.plot(d['prices'],mean,color=COLORS[h],ls=ls,marker='o',ms=2.5,label=label)
  stats[h]={'expected_price':float(mean@d['prices']),'probabilities':mean.tolist()}
 ax.axvline(1.475,color='#888888',ls='--',lw=.6,zorder=0)
 ax.axvline(1.925,color='#888888',ls=':',lw=.6,zorder=0)
 ax.set(xlim=(1.23,2.02),ylim=(0,1),xticks=[1.25,1.475,1.7,1.925],xticklabels=['1.25','1.475','1.70','1.925'],xlabel='Available price',title=f"{'a'if j==0 else'b'}  {name}")
 clean(ax);EVIDENCE['models'][name]['history']=stats
axes[0].set_ylabel('Action probability')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.015),ncol=3,frameon=False,columnspacing=1.25,handlelength=2)
save(fig,'history')

# Figure 3: matched parent interventions. Bootstrap complete trajectories.
fig,axes=plt.subplots(1,2,figsize=(5.5,2.5),sharey=True)
fig.subplots_adjust(left=.14,right=.985,bottom=.22,top=.75,wspace=.17)
for j,(name,d) in enumerate(DATA.items()):
 ax=axes[j];T=d['manifest']['rounds'];H=d['manifest']['branch_rounds'];B=T-H
 stats={}
 for kind,label in [('one_round_undercut','One-round price cut'),('competitive_rival','Fixed competitive rival')]:
  trajectories=[]
  for mid,entries in sorted(d['branches'].items()):
   if next(iter(entries.values()))['branch']!=kind:continue
   parent=next(iter(entries.values()))['parent_id']
   trajectories.append([np.dot(entries[t]['probabilities'],d['prices'])-np.dot(d['raw'][(parent,t)]['probabilities'],d['prices'])for t in range(B,T)])
  x=np.array(trajectories);mean,ci=boot(x)
  assert np.max(np.abs(x[:,0]))<1e-4 # Equal inputs; retain tiny batch-numerical differences.
  t=np.arange(H)
  ax.fill_between(t,*ci,color=COLORS[kind],alpha=.13,lw=0)
  ax.plot(t,mean,color=COLORS[kind],label=label)
  stats[kind]={'mean':mean.tolist(),'pointwise_ci95':ci.tolist(),'n':len(x)}
 ax.axhline(0,color='#454545',ls='--',lw=.8)
 ax.set(xlim=(0,H-1),ylim=(-.065,.026),xlabel='Rounds after intervention',title=f"{'a'if j==0 else'b'}  {name}")
 ax.xaxis.set_major_locator(MultipleLocator(5));clean(ax)
 EVIDENCE['models'][name]['branch_timecourses']=stats
axes[0].set_ylabel('Change in expected price')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.01),ncol=2,frameon=False,columnspacing=1.5,handlelength=2)
save(fig,'interventions')

# Figure 4: presentation sensitivity and one-shot decision quality.
fig,axes=plt.subplots(1,2,figsize=(5.5,2.5))
fig.subplots_adjust(left=.1,right=.985,bottom=.23,top=.84,wspace=.39)
ax=axes[0]
for i,(name,d)in enumerate(DATA.items()):
 vals=[]
 for firm in [0,1]:
  vals.append([next(r['expected_price']for r in d['diag']if r['type']=='menu_order'and r['history']=='empty'and r['firm']==firm and r['reverse']==rev)for rev in [False,True]])
  ax.plot([i-.22,i+.22],vals[-1],color=COLORS[name],alpha=.7,lw=.8,marker='o'if firm==0 else's',ms=4,mfc='white'if firm else COLORS[name])
 EVIDENCE['models'][name]['empty_menu_order']={'ascending_mean':float(np.array(vals)[:,0].mean()),'reversed_mean':float(np.array(vals)[:,1].mean())}
ax.set(xticks=[-.22,.22,.78,1.22],xticklabels=['A–K','K–A','A–K','K–A'],ylim=(1.45,1.85),ylabel='Expected next price',title='a  Empty-history menu order')
ax.text(0,.04,'Clef-flash',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8)
ax.text(1,.04,'Qwen3.5-9B',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8)
clean(ax)
ax=axes[1]
for name,d in DATA.items():
 pp=[r for r in d['diag']if r['type']=='one_shot'];assert len(pp)==22
 regrets=[np.mean([r['expected_profit_regret']for r in pp if r['rival_index']==i])for i in range(11)]
 acc=sum(r['argmax_correct']for r in pp)
 ax.plot(d['prices'],regrets,color=COLORS[name],marker='o',ms=2.5,label=f'{"Clef" if name=="Clef-flash" else "Qwen"}: {acc}/22 best choices')
 EVIDENCE['models'][name]['one_shot']={'argmax_correct':int(acc),'cases':22,'mean_expected_profit_regret':float(np.mean(regrets)),'regret_by_rival_price':regrets}
market=Market()
uniform_regret=[]
for rival in range(11):
 role_regret=[]
 for firm in [0,1]:
  pay=np.array([market.outcome([a,rival]if firm==0 else[rival,a])['profits'][firm]for a in range(11)])
  role_regret.append(float(pay.max()-pay.mean()))
 uniform_regret.append(float(np.mean(role_regret)))
ax.plot(d['prices'],uniform_regret,color='#777777',ls=':',lw=1.25,label='Uniform actions')
EVIDENCE['uniform_one_shot_regret_by_rival_price']=uniform_regret
ax.set(xlim=(1.23,2.02),xticks=[1.25,1.5,1.75,2],ylim=(0,.09),xlabel='Committed rival price',ylabel='Profit shortfall',title='b  Single-decision pricing')
ax.legend(frameon=False,loc='upper left',fontsize=7.4,handlelength=1.2);clean(ax)
save(fig,'diagnostics')

# Figure 5: scripted positive/negative controls; not empirical agent outcomes.
d=next(iter(DATA.values()));controls=read(d['path']/'final/scripted_controls.json')
fig,axes=plt.subplots(1,2,figsize=(5.5,2.35))
fig.subplots_adjust(left=.1,right=.985,bottom=.21,top=.76,wspace=.35)
for kind,label,color,ls in [('static_high','Static high','#8C8C8C',':'),('trigger','Trigger control','#BB6A35','-'),('nash','Nash control','#245C85','--')]:
 hist=controls[kind];t=np.array([r['round']+1 for r in hist]);p=np.array([r['prices'][0]for r in hist]);profit=np.array([r['profits'][0]for r in hist]);bench=d['summary']['benchmarks'];pn=bench['nash']['profits'][0];pj=bench['joint']['profits'][0]
 axes[0].plot(t,p,label=label,color=color,ls=ls)
 axes[1].plot(t,(profit-pn)/(pj-pn),label=label,color=color,ls=ls)
for ax,title in zip(axes,["a  Tracked firm's price","b  Tracked firm's profit gain"]):
 ax.axvline(31,color='#777777',lw=.7,ls=':');ax.set(xlim=(1,60),xlabel='Round',title=title);clean(ax)
axes[0].set(ylim=(1.42,1.98),ylabel='Price');axes[1].set(ylim=(-.1,1.1),ylabel='Profit gain $G$')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,1.01),ncol=3,frameon=False,handlelength=2)
save(fig,'controls')
(OUT/'figure_data.json').write_text(json.dumps(EVIDENCE,indent=2))
print(json.dumps({'output_pdfs':[str(p.relative_to(ROOT))for p in OUT.glob('*.pdf')],'previews':str(PREVIEW),'models':{n:{'history':{h:v['expected_price']for h,v in EVIDENCE['models'][n]['history'].items()},'one_shot':EVIDENCE['models'][n]['one_shot']['mean_expected_profit_regret']}for n in DATA}},indent=2))
