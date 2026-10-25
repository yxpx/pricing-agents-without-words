"""Exact CPU economics for a finite differentiated-product pricing game."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Market:
    prices: tuple[float, ...] = tuple(round(1.25 + .075*i, 6) for i in range(11))
    qualities: tuple[float, float] = (2., 2.)
    costs: tuple[float, float] = (1., 1.)
    mu: float = .25
    outside: float = 0.

    def outcome(self, actions):
        p=np.array([self.prices[int(a)] for a in actions],dtype=float)
        utilities=np.r_[self.outside,(np.asarray(self.qualities)-p)]/self.mu
        shift=float(utilities.max())
        logden=shift+math.log(float(np.exp(utilities-shift).sum()))
        q=np.exp(utilities[1:]-logden)
        profits=(p-np.asarray(self.costs))*q
        return {'prices':p.tolist(),'quantities':q.tolist(),'profits':profits.tolist(),
                'outside_share':float(math.exp(utilities[0]-logden)),
                'consumer_surplus':float(self.mu*logden-self.outside)}

    def payoff_matrix(self):
        k=len(self.prices)
        return np.array([[self.outcome([i,j])['profits'] for j in range(k)]for i in range(k)])

    def benchmarks(self):
        u=self.payoff_matrix();k=len(self.prices)
        nash=[(i,j)for i in range(k)for j in range(k)
              if u[i,j,0]>=u[:,j,0].max()-1e-10 and u[i,j,1]>=u[i,:,1].max()-1e-10]
        if len(nash)!=1:
            raise ValueError(f'Requires one unambiguous discrete pure Nash; found {nash}')
        joint=u.sum(-1);mx=float(joint.max())
        opt=[(i,j)for i in range(k)for j in range(k)if abs(joint[i,j]-mx)<1e-10]
        chosen=next((x for x in opt if x[0]==x[1]),opt[0])
        return {'grid':list(self.prices),'pure_nash_indices':[list(x) for x in nash],
                'nash_actions':list(nash[0]),'nash':self.outcome(nash[0]),
                'joint_optima':[list(x)for x in opt],'joint_actions':list(chosen),
                'joint':self.outcome(chosen)}

    def best_response(self,rival_action,firm=0):
        k=len(self.prices)
        vals=[self.outcome([a,rival_action]if firm==0 else[rival_action,a])['profits'][firm]for a in range(k)]
        return int(np.argmax(vals))
