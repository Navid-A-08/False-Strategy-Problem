"""
Experiment 8: real-data case study on the S&P 500 index (daily closes, 1990-2026, Yahoo Finance ^GSPC,
price index without dividends).

The 156 moving-average crossover rules are applied as long/flat timing rules (long the index when
the fast average is above the slow one, otherwise in cash; cash earns zero). Costs: 2 bps per unit
of turnover. The rules are selected on 1990-2010 and evaluated on 2011-2026. Reported: best rule's
in-sample and out-of-sample Sharpe ratios, its out-of-sample rank, buy-and-hold for comparison,
DSR with K estimated four ways, and the PBO of the whole search (CSCV, 16 blocks, full sample).
Output: results/exp8.json
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, spearmanr
from common import (ann_sr, ma_positions, cscv_pbo, z_fst, psr, PAIRS,
                    k_participation, k_liji, k_cluster)

COST = 2.0
px = pd.read_csv('results/sp500_daily.csv', index_col=0, parse_dates=True).iloc[:, 0].dropna()
p = np.log(px.values)
dates = px.index[1:]
P, ret = ma_positions(p)
P = (P > 0).astype(float)                                  # long / flat
turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0))
R = P * ret[:, None] - turn * COST / 1e4
start = 200                                                # all averages defined
R, ret, dates = R[start:], ret[start:], dates[start:]
split = np.searchsorted(dates, pd.Timestamp('2011-01-01'))
Ris, Ros = R[:split], R[split:]
sis, sos = ann_sr(Ris), ann_sr(Ros)
j = int(np.argmax(sis))
T = Ris.shape[0]
x = Ris[:, j]
C = np.corrcoef(Ris, rowvar=False)
Ks = dict(raw=float(len(PAIRS)), participation=k_participation(C), liji=k_liji(C), cluster=k_cluster(C))
s_pp = sis[j] / np.sqrt(252)
V = np.var(sis / np.sqrt(252), ddof=1)
dsr = {k: float(psr(s_pp, np.sqrt(V) * float(z_fst(v)), T, skew(x), kurtosis(x, fisher=False))) for k, v in Ks.items()}
dsr_null_var = {k: float(psr(s_pp, np.sqrt(1 / T) * float(z_fst(v)), T, skew(x), kurtosis(x, fisher=False))) for k, v in Ks.items()}
out = dict(
    period_is=[str(dates[0].date()), str(dates[split - 1].date())], period_oos=[str(dates[split].date()), str(dates[-1].date())],
    n_rules=len(PAIRS), best_rule=dict(fast=PAIRS[j][0], slow=PAIRS[j][1]),
    best_is_sr=float(sis[j]), best_oos_sr=float(sos[j]),
    best_oos_rank_pct=float((sos < sos[j]).mean()),
    median_is_sr=float(np.median(sis)), median_oos_sr=float(np.median(sos)),
    bh_is_sr=float(ann_sr(ret[:split])), bh_oos_sr=float(ann_sr(ret[split:])),
    share_rules_beating_bh_is=float((sis > ann_sr(ret[:split])).mean()),
    share_rules_beating_bh_oos=float((sos > ann_sr(ret[split:])).mean()),
    rankcorr_is_oos=float(spearmanr(sis, sos)[0]),
    naive_psr=float(psr(s_pp, 0, T, skew(x), kurtosis(x, fisher=False))),
    K=Ks, dsr_crosssec_var=dsr, dsr_null_var=dsr_null_var,
    pbo=cscv_pbo(R), cost_bps=COST)
json.dump(out, open('results/exp8.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
