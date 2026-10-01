"""
Experiment 7: how well can the effective number of independent trials be estimated, and what does
an error in it do to the Deflated Sharpe Ratio?

N = 400 zero-skill strategies are generated in M equal clusters with within-cluster correlation rho
(one common factor per cluster plus idiosyncratic noise). T = 1,260 days. For each replication:
  * the best strategy is selected on its Sharpe ratio;
  * K is estimated four ways: raw count N, eigenvalue participation ratio, the Li & Ji (2005)
    eigenvalue estimator, and hierarchical clustering with silhouette selection;
  * the DSR of the selected strategy is computed with each K (V = 1/T, the null sampling variance)
    and the strategy is declared significant if DSR > 0.95.
The "oracle" K is the K at which the False Strategy Theorem reproduces the simulated mean maximum.
A power scenario adds genuine skill (annualized SR 2.0) to one strategy.
Output: results/exp7.json
"""
import json
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq
from common import z_fst, psr, k_participation, k_liji, k_cluster

N, T, REPS = 400, 1260, 150
SCEN = [(5, 0.9), (20, 0.9), (80, 0.9), (5, 0.6), (20, 0.6), (80, 0.6)]
EST = ['raw', 'participation', 'liji', 'cluster']


def simulate(rng, M, rho, skill=0.0):
    size = N // M
    f = rng.standard_normal((T, M))
    e = rng.standard_normal((T, N))
    R = np.sqrt(rho) * np.repeat(f, size, axis=1) + np.sqrt(1 - rho) * e
    R *= 0.01
    if skill:
        R[:, 0] += skill / np.sqrt(252) * 0.01
    return R


def one_rep(seed, M, rho, skill):
    rng = np.random.default_rng(seed)
    R = simulate(rng, M, rho, skill)
    sr = R.mean(0) / R.std(0, ddof=1)
    j = int(np.argmax(sr))
    C = np.corrcoef(R, rowvar=False)
    K = dict(raw=float(N), participation=k_participation(C), liji=k_liji(C), cluster=k_cluster(C))
    sig = {k: bool(psr(sr[j], np.sqrt(1 / T) * float(z_fst(v)), T) > 0.95) for k, v in K.items()}
    return dict(max_sr=float(sr[j] * np.sqrt(252)), K=K, sig=sig, picked_skilled=(j == 0))


def summarise(reps, M, rho, skill):
    mx = np.mean([r['max_sr'] for r in reps])
    oracle = brentq(lambda K: np.sqrt(252 / T) * float(z_fst(K)) - mx, 1.01, 1e8)
    d = dict(M=M, rho=rho, skill=skill, mean_max_sr=float(mx), oracle_k=float(oracle),
             picked_skilled=float(np.mean([r['picked_skilled'] for r in reps])))
    for e in EST:
        ks = np.array([r['K'][e] for r in reps])
        d[f'k_{e}'] = float(np.median(ks)); d[f'k_{e}_p10'] = float(np.percentile(ks, 10)); d[f'k_{e}_p90'] = float(np.percentile(ks, 90))
        d[f'sig_{e}'] = float(np.mean([r['sig'][e] for r in reps]))
        if skill:
            d[f'power_{e}'] = float(np.mean([r['sig'][e] and r['picked_skilled'] for r in reps]))
    # DSR with the oracle K
    d['sig_oracle'] = float(np.mean([psr(r['max_sr'] / np.sqrt(252), np.sqrt(1 / T) * float(z_fst(oracle)), T) > 0.95 for r in reps]))
    return d


if __name__ == '__main__':
    out = []
    for (M, rho) in SCEN:
        reps = Parallel(n_jobs=2)(delayed(one_rep)(70000 + 1000 * M + int(rho * 10) * 100000 + i, M, rho, 0.0) for i in range(REPS))
        d = summarise(reps, M, rho, 0.0); out.append(d); print(d, flush=True)
    power = []
    for (M, rho) in [(20, 0.9), (80, 0.6)]:
        reps = Parallel(n_jobs=2)(delayed(one_rep)(990000 + 1000 * M + i, M, rho, 2.0) for i in range(REPS))
        d = summarise(reps, M, rho, 2.0); power.append(d); print(d, flush=True)
    json.dump(dict(null=out, power=power, N=N, T=T, reps=REPS), open('results/exp7.json', 'w'), indent=1)
