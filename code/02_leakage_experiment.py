"""
Experiment 4: cross-validation leakage with overlapping 20-day labels on a random walk.
Random forest accuracy under shuffled 5-fold CV, purged + embargoed 5-fold CV, and walk-forward testing.

Output: results/leak.json   Runtime: about 1-2 minutes.
"""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
rng = np.random.default_rng(7)
H = 20; reps = 10; N = 2500
res = {'shuffled': [], 'purged': [], 'walkforward': []}
def purged_folds(n, k, h, emb):
    idx = np.arange(n); folds = np.array_split(idx, k)
    for te in folds:
        lo, hi = te[0], te[-1]
        tr = idx[(idx < lo - h) | (idx > hi + h + emb)]
        yield tr, te
for _ in range(reps):
    r = rng.standard_normal(N + 300) * 0.01
    p = np.cumsum(r)
    t = np.arange(100, N + 100)
    X = np.column_stack([p[t] - p[t - L] for L in (5, 10, 20, 60)])
    X = np.column_stack([X, np.array([r[i-20:i].std() for i in t])])
    y = (p[t + H] - p[t] > 0).astype(int)
    def score(splits):
        acc = []
        for tr, te in splits:
            m = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
            acc.append((m.predict(X[te]) == y[te]).mean())
        return float(np.mean(acc))
    res['shuffled'].append(score(KFold(5, shuffle=True, random_state=0).split(X)))
    res['purged'].append(score(purged_folds(N, 5, H, 25)))
    wf = []
    for k in range(1, 5):
        cut = k * N // 5
        wf.append((np.arange(0, cut - H), np.arange(cut, min(cut + N // 5, N))))
    res['walkforward'].append(score(wf))
summ = {k: dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v))) for k, v in res.items()}
print(summ)
json.dump(summ, open('results/leak.json', 'w'))
