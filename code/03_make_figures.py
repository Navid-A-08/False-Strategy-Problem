"""
Draws Figures 1-4 of the paper from results/res.json and results/leak.json into figures/.
Paper figure numbering: fig2.pdf = Figure 1 (MinBTL), fig1.pdf = Figure 2 (Exp 1),
fig3.pdf = Figure 3 (PBO), fig4.pdf = Figure 4 (leakage).
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
D = ''
R = json.load(open('results/res.json')); L = json.load(open('results/leak.json'))
G = 0.5772156649; BLUE = '#2563a8'; GREY = '#8a8f98'; DARK = '#333333'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.labelcolor': DARK, 'xtick.color': '#555', 'ytick.color': '#555'})
def z(K): return (1-G)*norm.ppf(1-1/K) + G*norm.ppf(1-1/(K*np.e))

# Fig 1
e = R['exp1']; K = np.array([d['K'] for d in e])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
Ks = np.logspace(np.log10(2), 4, 200)
ax.plot(Ks, np.sqrt(252/1260)*z(Ks), color=GREY, lw=1.2, label='False Strategy Theorem prediction')
m = np.array([d['mean_best_is'] for d in e]); lo = np.array([d['p5'] for d in e]); hi = np.array([d['p95'] for d in e])
ax.errorbar(K, m, yerr=[m-lo, hi-m], fmt='o', color=BLUE, ms=5, capsize=3, label='Simulated best in-sample SR (mean, 5th–95th pct.)')
ax.plot(K, [d['mean_oos'] for d in e], 's', color=DARK, ms=4, mfc='white', label='Same strategy, out-of-sample SR (mean)')
ax.axhline(0, color='#bbb', lw=0.8); ax.axhline(1, color='#bbb', lw=0.8, ls='--')
ax.text(1.05, 1.03, 'SR = 1.0', color='#777', fontsize=7.5)
ax.set_xscale('log'); ax.set_xlabel('Number of independent strategies tried, K (log scale)'); ax.set_ylabel('Annualized Sharpe ratio')
ax.set_ylim(-0.9,2.3); ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.12,1.02)); fig.tight_layout(); fig.savefig('figures/fig1.pdf')

# Fig 2: MinBTL
fig, ax = plt.subplots(figsize=(5.2, 2.7))
Ks = np.logspace(np.log10(2), 4, 200)
ax.plot(Ks, z(Ks)**2, color=BLUE, lw=1.6, label='Target in-sample SR = 1.0')
ax.plot(Ks, (z(Ks)/0.5)**2, color=GREY, lw=1.2, label='Target in-sample SR = 0.5')
ax.axhline(5, color='#bbb', lw=0.8, ls='--'); ax.text(2.1, 5.8, '5 years of data', color='#777', fontsize=7.5)
ax.plot([45], [z(45)**2], 'o', color=BLUE, ms=4); ax.annotate('K ≈ 45', (45, z(45)**2), (90, 1.5), fontsize=7.5, color=DARK,
          arrowprops=dict(arrowstyle='-', color='#999', lw=0.7))
ax.set_xscale('log'); ax.set_ylim(0, 65)
ax.set_xlabel('Number of independent trials, K (log scale)'); ax.set_ylabel('Minimum backtest length (years)')
ax.legend(frameon=False, fontsize=7, loc='upper left'); fig.tight_layout(); fig.savefig('figures/fig2.pdf')

# Fig 3: PBO logits
h = R['exp2']['logit_hist']
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for b in h:
    ax.bar((b['lo']+b['hi'])/2, b['n'], width=0.46, color=(GREY if b['hi'] > 0 else BLUE))
ax.axvline(0, color=DARK, lw=0.8)
ax.text(-3.9, max(b['n'] for b in h)*0.92, 'Selected rule ranks below\nmedian out-of-sample', color=BLUE, fontsize=7.5)
ax.text(1.9, max(b['n'] for b in h)*0.92, 'Ranks above median', color='#666', fontsize=7.5)
ax.set_xlabel('Logit of selected rule\'s out-of-sample relative rank (λ)'); ax.set_ylabel('Number of CSCV splits')
fig.tight_layout(); fig.savefig('figures/fig3.pdf')

# Fig 4: leakage
fig, ax = plt.subplots(figsize=(5.2, 2.2))
names = ['Shuffled 5-fold CV', 'Purged + embargoed 5-fold CV', 'Walk-forward (with gap)']
keys = ['shuffled', 'purged', 'walkforward']
for i, k in enumerate(keys):
    v = L[k]; c = BLUE if k == 'shuffled' else GREY
    ax.barh(i, v['mean']*100, color=c, height=0.55)
    ax.errorbar(v['mean']*100, i, xerr=[[100*(v['mean']-v['lo'])], [100*(v['hi']-v['mean'])]], color=DARK, capsize=3, lw=0.8)
    ax.text(v['hi']*100+0.8, i, f"{v['mean']*100:.1f}%", va='center', fontsize=8, color=DARK)
ax.axvline(50, color=DARK, lw=0.8, ls='--'); ax.text(50.6, -0.45, 'coin flip (50%)', fontsize=7.5, color='#666')
ax.set_yticks(range(3)); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 75)
ax.set_xlabel('Directional accuracy on a pure random walk (%)')
fig.tight_layout(); fig.savefig('figures/fig4.pdf')
print('ok')
