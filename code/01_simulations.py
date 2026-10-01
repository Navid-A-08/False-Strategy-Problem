"""
Experiments 1-3 of "The False Strategy Problem in Machine Learning for Finance" (N. Abdollahzadeh, 2026).

  Exp 1  Best-of-K selection among K zero-skill strategies (5-year in-sample / out-of-sample),
         compared with the False Strategy Theorem prediction.
  MinBTL Minimum backtest length for K trials (Bailey et al., 2014).
  Exp 2  156 moving-average crossover rules on Gaussian random walks (3y in-sample, 2y out-of-sample),
         plus Probability of Backtest Overfitting via CSCV with 16 blocks.
  Exp 3  Naive PSR vs Deflated Sharpe Ratio for the selected strategy (Student-t(5) returns).

Output: results/res.json   Runtime: about 7 minutes on 2 CPU cores.
"""
import numpy as np, json, itertools
from scipy.stats import norm, skew, kurtosis
rng = np.random.default_rng(20261001)
G = 0.5772156649
out = {}

def emax_sr(K, sd):
    if K == 1: return 0.0
    return sd * ((1-G)*norm.ppf(1-1/K) + G*norm.ppf(1-1/(K*np.e)))

def ann_sr(r, axis=0):
    return r.mean(axis)/r.std(axis, ddof=1)*np.sqrt(252)

# ---------- Exp 1: pure-noise strategies ----------
T = 1260; Toos = 1260
sd_ann = np.sqrt(252/T)
exp1 = []
for K, reps in [(1,2000),(10,2000),(100,1000),(1000,300),(10000,40)]:
    best_is, best_oos = [], []
    for _ in range(reps):
        r = rng.standard_normal((T, K)).astype(np.float32)*0.01
        s = ann_sr(r)
        j = int(np.argmax(s)); best_is.append(s[j])
        best_oos.append(ann_sr(rng.standard_normal(Toos)*0.01))
    best_is = np.array(best_is)
    exp1.append(dict(K=K, mean_best_is=float(best_is.mean()), p5=float(np.percentile(best_is,5)),
                     p95=float(np.percentile(best_is,95)), fst=float(emax_sr(K, sd_ann)),
                     share_above_1=float((best_is>1).mean()), mean_oos=float(np.mean(best_oos))))
out['exp1'] = exp1

# ---------- MinBTL: years needed so E[max] of K noise trials stays below target SR ----------
def minbtl(K, target):
    z = (1-G)*norm.ppf(1-1/K) + G*norm.ppf(1-1/(K*np.e))
    return (z/target)**2
out['minbtl'] = [dict(K=K, years_sr1=float(minbtl(K,1.0)), years_sr05=float(minbtl(K,0.5))) for K in [10,45,100,1000,10000]]

# ---------- Exp 2: MA crossover grid on random walks ----------
fast = list(range(2, 51, 4)); slow = list(range(20, 201, 15))
pairs = [(f, s) for f in fast for s in slow if f < s]
def ma_strats(p):
    cs = np.concatenate([[0], np.cumsum(p)])
    def ma(n):
        m = np.full(len(p), np.nan); m[n-1:] = (cs[n:] - cs[:-n])/n; return m
    cache = {n: ma(n) for n in set(fast)|set(slow)}
    ret = np.diff(p)  # log-return t->t+1
    R = np.empty((len(ret), len(pairs)))
    for k, (f, s) in enumerate(pairs):
        pos = np.sign(cache[f] - cache[s])[:-1]
        R[:, k] = np.nan_to_num(pos) * ret
    return R
reps2 = 200; burn = 200; Tis = 756; To = 504
is_best, oos_best, rankcorr, pbo_logits = [], [], [], []
from scipy.stats import spearmanr
def cscv_pbo(R, S=16):
    n = (R.shape[0]//S)*S; R = R[:n]; blocks = np.array_split(np.arange(n), S)
    lam = []
    for comb in itertools.combinations(range(S), S//2):
        tr = np.concatenate([blocks[i] for i in comb]); te = np.setdiff1d(np.arange(n), tr)
        s_tr = R[tr].mean(0)/R[tr].std(0); s_te = R[te].mean(0)/R[te].std(0)
        j = np.argmax(s_tr); w = (s_te < s_te[j]).mean() + 0.5/len(s_te)  # relative rank
        w = min(max(w, 1e-6), 1-1e-6); lam.append(np.log(w/(1-w)))
    lam = np.array(lam); return float((lam <= 0).mean()), lam
pbos = []; lam_example = None
for rep in range(reps2):
    p = np.cumsum(rng.standard_normal(burn+Tis+To+1)*0.01)
    R = ma_strats(p)[burn-1:]
    Ris, Ros = R[:Tis], R[Tis:Tis+To]
    sis, sos = ann_sr(Ris), ann_sr(Ros)
    j = int(np.argmax(sis)); is_best.append(sis[j]); oos_best.append(sos[j])
    rankcorr.append(spearmanr(sis, sos)[0])
    if rep < 20:
        pb, lam = cscv_pbo(R[:Tis+To]); pbos.append(pb)
        if lam_example is None: lam_example = lam
out['exp2'] = dict(n_rules=len(pairs), mean_best_is=float(np.mean(is_best)), mean_best_oos=float(np.mean(oos_best)),
    share_is_above_1=float((np.array(is_best)>1).mean()), share_oos_positive=float((np.array(oos_best)>0).mean()),
    mean_rankcorr=float(np.mean(rankcorr)), pbo_mean=float(np.mean(pbos)), pbo_min=float(np.min(pbos)), pbo_max=float(np.max(pbos)),
    fst_effective=None)
hist, edges = np.histogram(lam_example, bins=np.arange(-4, 4.01, 0.5))
out['exp2']['logit_hist'] = [dict(lo=float(edges[i]), hi=float(edges[i+1]), n=int(hist[i])) for i in range(len(hist))]
out['exp2']['logit_clipped'] = int(((lam_example < -4) | (lam_example > 4)).sum())

# ---------- Exp 3: Deflated Sharpe Ratio ----------
def psr(sr_hat, sr0, T, g3, g4):  # per-period SR
    return norm.cdf((sr_hat - sr0)*np.sqrt(T-1)/np.sqrt(1 - g3*sr_hat + (g4-1)/4*sr_hat**2))
def dsr_case(K, true_sr_ann, reps=200, T=1260):
    res = []
    for _ in range(reps):
        r = rng.standard_t(5, size=(T, K)).astype(np.float32)/np.sqrt(5/3)*0.01
        mu = true_sr_ann/np.sqrt(252)*0.01; r[:, 0] += mu
        s = r.mean(0)/r.std(0, ddof=1)
        j = int(np.argmax(s)); x = r[:, j].astype(float)
        g3 = skew(x); g4 = kurtosis(x, fisher=False)
        sr0 = emax_sr(K, np.std(s, ddof=1))
        res.append(dict(naive=psr(s[j], 0, T, g3, g4), dsr=psr(s[j], sr0, T, g3, g4), picked_true=(j==0)))
    return dict(K=K, true_sr=true_sr_ann,
        naive_reject=float(np.mean([d['naive']>0.95 for d in res])),
        dsr_reject=float(np.mean([d['dsr']>0.95 for d in res])),
        picked_true=float(np.mean([d['picked_true'] for d in res])))
out['exp3'] = [dsr_case(1000, 0.0), dsr_case(1000, 1.0), dsr_case(1000, 2.0), dsr_case(100, 1.0)]

json.dump(out, open('results/res.json','w'), indent=1)
print(json.dumps({k:v for k,v in out.items()}, indent=1, default=str)[:6000])
