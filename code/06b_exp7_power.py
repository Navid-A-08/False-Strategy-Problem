"""
Experiment 7b: power of the DSR under each effective-K estimator, for weak to strong genuine signals.

Same design as Experiment 7 (N = 400 strategies in M clusters, within-cluster correlation rho,
T = 1,260 days). One strategy is given a true annualized Sharpe ratio of 0.5, 1.0, 1.5 or 2.0.
Power = share of replications in which the skilled strategy is the one selected AND its DSR > 0.95.
The oracle K for each scenario is taken from the zero-skill runs of Experiment 7 (results/exp7.json).
Output: results/exp7b.json
"""
import json
import numpy as np
from joblib import Parallel, delayed
from common import z_fst, psr
import importlib
exp7 = importlib.import_module("06_exp7_effective_k")
one_rep, SCEN, EST, T = exp7.one_rep, exp7.SCEN, exp7.EST, exp7.T
# from exp7_effective_k import one_rep, SCEN, EST, T

REPS = 120
SKILLS = [0.5, 1.0, 1.5, 2.0]

if __name__ == '__main__':
    oracle = {(s['M'], s['rho']): s['oracle_k'] for s in json.load(open('results/exp7.json'))['null']}
    out = []
    for (M, rho) in SCEN:
        for sk in SKILLS:
            seeds = [3_000_000 + 100_000 * int(rho * 10) + 1000 * M + int(sk * 10) * 7919 + i for i in range(REPS)]
            reps = Parallel(n_jobs=2)(delayed(one_rep)(sd, M, rho, sk) for sd in seeds)
            ko = oracle[(M, rho)]
            d = dict(M=M, rho=rho, skill=sk, selected_skilled=float(np.mean([r['picked_skilled'] for r in reps])))
            for e in EST:
                d[f'power_{e}'] = float(np.mean([r['sig'][e] and r['picked_skilled'] for r in reps]))
            d['power_oracle'] = float(np.mean([r['picked_skilled'] and
                                               psr(r['max_sr'] / np.sqrt(252), np.sqrt(1 / T) * float(z_fst(ko)), T) > 0.95
                                               for r in reps]))
            out.append(d)
            print(d, flush=True)
    json.dump(dict(rows=out, reps=REPS), open('results/exp7b.json', 'w'), indent=1)
