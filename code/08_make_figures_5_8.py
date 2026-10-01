"""Figures 5-7 for the extended experiments."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import z_fst, ann_sr, ma_positions, PAIRS

BLUE, GREY, DARK, LIGHT = '#2563a8', '#8a8f98', '#333333', '#c9ccd1'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.labelcolor': DARK, 'xtick.color': '#555', 'ytick.color': '#555'})

# ---------- Figure 5: ML search ----------
d = json.load(open('results/exp5.json'))['curve']
K = np.array([c['K'] for c in d])
fig, ax = plt.subplots(figsize=(5.4, 3.0))
Ks = np.logspace(np.log10(2), np.log10(200), 100)
ax.plot(Ks, z_fst(Ks), color=GREY, lw=1.1, ls='--', label='FST prediction if configurations were independent')
v = np.array([c['best_val_sr'] for c in d]); lo = np.array([c['val_p10'] for c in d]); hi = np.array([c['val_p90'] for c in d])
ax.errorbar(K, v, yerr=[v - lo, hi - v], fmt='o', color=BLUE, ms=5, capsize=3, label='Selected model: validation SR (mean, 10th–90th pct.)')
ax.plot(K, [c['test_sr'] for c in d], 's', color=DARK, ms=4, mfc='white', label='Same model: untouched test-year SR (mean)')
ax.axhline(0, color='#bbb', lw=0.8)
ax.set_xscale('log'); ax.set_ylim(-1.6, 4.3)
ax.set_xlabel('Number of model configurations tried, K (log scale)'); ax.set_ylabel('Annualized Sharpe ratio')
ax.legend(frameon=False, fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig('figures/fig5.pdf')

# ---------- Figure 6: effective K ----------
e = json.load(open('results/exp7.json'))['null']
labels = [f"M={s['M']}\nρ={s['rho']}" for s in e]
est = [('raw', 'Raw count', LIGHT, 's'), ('liji', 'Li–Ji eigenvalue', BLUE, 'o'),
       ('participation', 'Participation ratio', GREY, '^'), ('cluster', 'Cluster count', DARK, 'D')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
x = np.arange(len(e))
for k, (key, name, col, mk) in enumerate(est):
    ratio = [s[f'k_{key}'] / s['oracle_k'] for s in e]
    a1.plot(x + (k - 1.5) * 0.13, ratio, mk, color=col, ms=5, label=name, mec=DARK if col == LIGHT else col)
a1.axhline(1, color=DARK, lw=0.8); a1.text(-0.45, 1.1, 'correct K', fontsize=7, color='#555')
a1.set_yscale('log'); a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7)
a1.set_ylabel('Estimated K ÷ oracle K (log scale)'); a1.set_title('(a) Accuracy of effective-K estimates', fontsize=9)
h, l = a1.get_legend_handles_labels()
for k, (key, name, col, mk) in enumerate(est):
    fpr = [100 * s[f'sig_{key}'] for s in e]
    a2.bar(x + (k - 1.5) * 0.2, fpr, width=0.19, color=col, edgecolor=DARK if col == LIGHT else col, lw=0.5)
a2.axhline(5, color=DARK, lw=0.8, ls='--'); a2.text(5.5, 5.6, 'nominal 5%', fontsize=7, color='#555', ha='right')
a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7)
a2.legend(h, l, frameon=False, fontsize=6.8, loc='upper left', bbox_to_anchor=(0.0, 0.93))
a2.set_ylabel('False-positive rate of DSR test (%)'); a2.set_title('(b) Zero-skill winners declared significant', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig6.pdf')

# ---------- Figure 7: S&P 500 case ----------
import os
if not os.path.exists('results/sp500_daily.csv'):
    print('S&P 500 data not found: run 00_download_sp500.py to draw fig7.pdf; skipping it')
else:
    r8 = json.load(open('results/exp8.json'))
    px = pd.read_csv('results/sp500_daily.csv', index_col=0, parse_dates=True).iloc[:, 0].dropna()
    p = np.log(px.values); dates = px.index[1:]
    P, ret = ma_positions(p); P = (P > 0).astype(float)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0))
    R = (P * ret[:, None] - turn * 2.0 / 1e4)[200:]; ret = ret[200:]; dates = dates[200:]
    split = np.searchsorted(dates, pd.Timestamp('2011-01-01'))
    sis, sos = ann_sr(R[:split]), ann_sr(R[split:])
    j = int(np.argmax(sis))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(sis, sos, 'o', color=GREY, ms=3.5, alpha=0.8, mec='none', label='156 crossover rules (long/flat, net of costs)')
    ax.plot(sis[j], sos[j], 'o', color=BLUE, ms=7, label=f'In-sample winner ({PAIRS[j][0]}/{PAIRS[j][1]}-day)')
    ax.axvline(r8['bh_is_sr'], color=LIGHT, lw=0.9, ls='--'); ax.axhline(r8['bh_oos_sr'], color=LIGHT, lw=0.9, ls='--')
    ax.text(r8['bh_is_sr'] + 0.005, max(sos) + 0.005, 'buy-and-hold 1990–2010', fontsize=7, color='#666', va='bottom')
    ax.text(max(sis) + 0.01, r8['bh_oos_sr'] + 0.01, 'buy-and-hold 2011–2026', fontsize=7, color='#666', ha='right', va='bottom')
    ax.set_xlabel('Sharpe ratio, 1990–2010 (selection period)'); ax.set_ylabel('Sharpe ratio, 2011–2026 (evaluation)')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout(); fig.savefig('figures/fig7.pdf')
    print('ok')

# ---------- Figure 8: power by signal strength ----------
P = json.load(open('results/exp7b.json'))['rows']
N0 = json.load(open('results/exp7.json'))['null']
sks = [0.5, 1.0, 1.5, 2.0]
fig, ax = plt.subplots(figsize=(5.4, 3.1))
series = [('raw', 'Raw count', LIGHT, 's', '-'), ('liji', 'Li–Ji eigenvalue', BLUE, 'o', '-'),
          ('participation', 'Participation ratio', GREY, '^', '-'), ('cluster', 'Cluster count', DARK, 'D', '-'),
          ('oracle', 'Oracle K', DARK, None, ':')]
for key, name, col, mk, ls in series:
    pw = [100 * np.mean([r[f'power_{key}'] for r in P if r['skill'] == s]) for s in sks]
    fpr = 100 * np.mean([s[f'sig_{key}'] for s in N0])
    ax.plot(sks, pw, ls, color=col, marker=mk, ms=5, lw=1.4 if key == 'liji' else 1.1,
            mec=DARK if col == LIGHT else col, label=f'{name} (false-positive rate {fpr:.1f}%)')
sel = [100 * np.mean([r['selected_skilled'] for r in P if r['skill'] == s]) for s in sks]
ax.plot(sks, sel, color='#bbb', lw=0.9, ls='--')
ax.text(1.62, 71, 'skilled strategy\nselected (any K)', fontsize=7, color='#777', ha='left')
ax.set_xticks(sks); ax.set_xlim(0.4, 2.1); ax.set_ylim(0, 100)
ax.set_xlabel('True annualized Sharpe ratio of the one skilled strategy'); ax.set_ylabel('Power: skilled strategy selected\nand DSR > 0.95 (%)')
ax.legend(frameon=False, fontsize=6.8, loc='upper left')
fig.tight_layout(); fig.savefig('figures/fig8.pdf')
print('fig8 ok')
