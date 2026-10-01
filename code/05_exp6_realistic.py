"""
Experiment 6: robustness of Experiment 2 to realistic market dynamics.

Price paths have zero expected return but realistic volatility: GARCH(1,1) with Student-t(5)
innovations, plus a two-state Markov volatility regime (calm / turbulent). The same 156
moving-average crossover rules are searched over 3 years and evaluated over the next 2 years,
before and after transaction costs (2 bps per unit of turnover; a long-to-short flip costs 4 bps).
Gaussian random walks (Experiment 2 design) are rerun alongside as the baseline.
Output: results/exp6.json
"""
import json
import numpy as np
from scipy.stats import spearmanr, kurtosis
from common import ann_sr, ma_positions, rule_returns, cscv_pbo, z_fst

N_PATHS, N_PBO, BURN, TIS, TOS, COST = 200, 20, 200, 756, 504, 2.0


def garch_regime_path(rng, n):
    omega, alpha, beta, nu = 1e-6, 0.09, 0.90, 5.0               # daily variance units
    calm, turb, p_cc, p_tt = 0.8, 1.6, 0.995, 0.98
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    h = omega / (1 - alpha - beta)
    s = 0
    r = np.empty(n)
    for t in range(n):
        if s == 0 and rng.random() > p_cc: s = 1
        elif s == 1 and rng.random() > p_tt: s = 0
        mult = calm if s == 0 else turb
        r[t] = np.sqrt(h) * mult * z[t]
        h = omega + alpha * (r[t] / mult) ** 2 + beta * h
    return r / r.std() * 0.01           # rescale to 1% daily volatility on average


def run(kind, rng):
    is_g, oos_g, is_n, oos_n, rc, pbos, kurt, acf_sq = [], [], [], [], [], [], [], []
    for i in range(N_PATHS):
        n = BURN + TIS + TOS + 1
        r = garch_regime_path(rng, n) if kind == 'garch' else rng.standard_normal(n) * 0.01
        if kind == 'garch':
            kurt.append(kurtosis(r)); acf_sq.append(np.corrcoef(r[1:] ** 2, r[:-1] ** 2)[0, 1])
        p = np.concatenate([[0], np.cumsum(r)])
        P, ret = ma_positions(p)
        P, ret = P[BURN - 1:], ret[BURN - 1:]
        Rg = rule_returns(P, ret)
        Rn = rule_returns(P, ret, COST)
        sg_is, sg_os = ann_sr(Rg[:TIS]), ann_sr(Rg[TIS:TIS + TOS])
        sn_is, sn_os = ann_sr(Rn[:TIS]), ann_sr(Rn[TIS:TIS + TOS])
        j = int(np.argmax(sg_is))           # researcher selects on gross in-sample Sharpe
        is_g.append(sg_is[j]); oos_g.append(sg_os[j]); is_n.append(sn_is[j]); oos_n.append(sn_os[j])
        rc.append(spearmanr(sg_is, sg_os)[0])
        if i < N_PBO:
            pbos.append(cscv_pbo(Rn[:TIS + TOS]))
    out = dict(mean_best_is_gross=float(np.mean(is_g)), mean_best_is_net=float(np.mean(is_n)),
               mean_oos_gross=float(np.mean(oos_g)), mean_oos_net=float(np.mean(oos_n)),
               share_is_gt1=float(np.mean(np.array(is_g) > 1)), share_oos_net_pos=float(np.mean(np.array(oos_n) > 0)),
               p90_best_is=float(np.percentile(is_g, 90)),
               rankcorr=float(np.mean(rc)), pbo_mean=float(np.mean(pbos)),
               k_eff=None)
    if kind == 'garch':
        out['excess_kurtosis'] = float(np.mean(kurt)); out['acf_sq_lag1'] = float(np.mean(acf_sq))
    from scipy.optimize import brentq
    target = out['mean_best_is_gross']
    out['k_eff'] = float(brentq(lambda K: np.sqrt(252 / TIS) * float(z_fst(K)) - target, 1.01, 1e7))
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(6006)
    res = dict(gaussian=run('gauss', rng), garch=run('garch', rng), cost_bps=COST)
    json.dump(res, open('results/exp6.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
