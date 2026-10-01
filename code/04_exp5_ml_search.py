"""
Experiment 5: hyperparameter / feature / seed search with ML models on unpredictable data.

Each path is a Gaussian random walk (no predictability). Next-day direction is predicted from
12 candidate technical features. Data are split in time: 3 years train, 1 year validation,
1 year test. 200 random configurations (model family, hyperparameters, feature subset, seed)
are fitted per path; the configuration with the best validation Sharpe ratio is "selected"
and evaluated on the untouched test year.
Output: results/exp5.json
"""
import json, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from common import ann_sr, z_fst, psr
warnings.filterwarnings('ignore')

N_PATHS, N_CFG = 24, 200
TR, VA, TE, WARM = 756, 252, 252, 100


def features(r, p):
    t = np.arange(WARM, len(r) - 1)
    F = [r[t - L] for L in range(0, 5)]                      # last 5 daily returns
    F += [p[t] - p[t - L] for L in (5, 10, 20, 60)]          # momentum
    F += [np.array([r[i - L + 1:i + 1].std() for i in t]) for L in (5, 20)]  # volatility
    F += [p[t] - np.array([p[i - 19:i + 1].mean() for i in t])]              # distance to 20d MA
    X = np.column_stack(F)
    y = (r[t + 1] > 0).astype(int)
    fwd = r[t + 1]
    return X, y, fwd


def random_config(rng):
    fam = rng.choice(['gbm', 'rf', 'logit', 'mlp'], p=[0.35, 0.3, 0.15, 0.2])
    nf = int(rng.integers(3, 13))
    feats = np.sort(rng.choice(12, nf, replace=False))
    seed = int(rng.integers(0, 10 ** 6))
    if fam == 'gbm':
        hp = dict(learning_rate=float(rng.choice([0.01, 0.03, 0.1, 0.3])),
                  max_depth=int(rng.choice([2, 3, 4, 6])), max_iter=int(rng.choice([50, 100, 200])),
                  min_samples_leaf=int(rng.choice([5, 20, 50])))
    elif fam == 'rf':
        hp = dict(n_estimators=60, max_depth=int(rng.choice([2, 4, 8, 16])),
                  min_samples_leaf=int(rng.choice([1, 5, 20, 50])), max_features=float(rng.choice([0.3, 0.6, 1.0])))
    elif fam == 'logit':
        hp = dict(C=float(rng.choice([0.001, 0.01, 0.1, 1, 10])))
    else:
        hp = dict(hidden_layer_sizes=(int(rng.choice([8, 16, 32])),), alpha=float(rng.choice([1e-4, 1e-2, 1])),
                  max_iter=300)
    return fam, feats, seed, hp


def make(fam, seed, hp):
    if fam == 'gbm':
        return HistGradientBoostingClassifier(random_state=seed, **hp)
    if fam == 'rf':
        return RandomForestClassifier(random_state=seed, n_jobs=1, **hp)
    if fam == 'logit':
        return LogisticRegression(**hp)
    return MLPClassifier(random_state=seed, **hp)


def one_path(i):
    rng = np.random.default_rng(5000 + i)
    n = WARM + TR + VA + TE + 2
    r = rng.standard_normal(n) * 0.01
    p = np.cumsum(r)
    X, y, fwd = features(r, p)
    a, b = TR, TR + VA
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-12
    Xs = (X - mu) / sd
    rows = []
    for c in range(N_CFG):
        fam, feats, seed, hp = random_config(rng)
        m = make(fam, seed, hp).fit(Xs[:a][:, feats], y[:a])
        pred = m.predict(Xs[a:][:, feats])
        pos = 2 * pred - 1
        strat = pos * fwd[a:]
        rows.append(dict(fam=str(fam), val_sr=float(ann_sr(strat[:VA])), test_sr=float(ann_sr(strat[VA:VA + TE])),
                         val_acc=float((pred[:VA] == y[a:b]).mean()), test_acc=float((pred[VA:VA + TE] == y[b:b + TE]).mean())))
    return rows


if __name__ == '__main__':
    allrows = Parallel(n_jobs=2)(delayed(one_path)(i) for i in range(N_PATHS))
    Ks = [1, 5, 20, 50, 100, 200]
    rng = np.random.default_rng(55)
    curve = []
    for K in Ks:
        bv, bt, ba, bta = [], [], [], []
        for rows in allrows:
            for _ in range(20 if K < N_CFG else 1):     # random subsets of size K
                idx = rng.choice(N_CFG, K, replace=False)
                j = max(idx, key=lambda q: rows[q]['val_sr'])
                bv.append(rows[j]['val_sr']); bt.append(rows[j]['test_sr'])
                ba.append(rows[j]['val_acc']); bta.append(rows[j]['test_acc'])
        curve.append(dict(K=K, best_val_sr=float(np.mean(bv)), test_sr=float(np.mean(bt)),
                          best_val_acc=float(np.mean(ba)), test_acc=float(np.mean(bta)),
                          val_p10=float(np.percentile(bv, 10)), val_p90=float(np.percentile(bv, 90)),
                          test_p10=float(np.percentile(bt, 10)), test_p90=float(np.percentile(bt, 90)),
                          fst=float(np.sqrt(252 / VA) * z_fst(K))))
    # DSR of the selected configuration (K = 200 trials, V = 1/T per period)
    dsr_sig, naive_sig, sel_val, sel_test, share_gt1 = 0, 0, [], [], 0
    for rows in allrows:
        j = int(np.argmax([q['val_sr'] for q in rows]))
        s = rows[j]['val_sr'] / np.sqrt(252)
        naive_sig += psr(s, 0, VA) > 0.95
        dsr_sig += psr(s, np.sqrt(1 / VA) * z_fst(N_CFG), VA) > 0.95
        sel_val.append(rows[j]['val_sr']); sel_test.append(rows[j]['test_sr'])
        share_gt1 += rows[j]['val_sr'] > 1
    fams = {}
    for rows in allrows:
        j = int(np.argmax([q['val_sr'] for q in rows]))
        fams[rows[j]['fam']] = fams.get(rows[j]['fam'], 0) + 1
    out = dict(curve=curve, n_paths=N_PATHS, n_cfg=N_CFG,
               selected=dict(mean_val_sr=float(np.mean(sel_val)), mean_test_sr=float(np.mean(sel_test)),
                             share_val_sr_gt1=share_gt1 / N_PATHS, naive_sig=naive_sig / N_PATHS,
                             dsr_sig=dsr_sig / N_PATHS, test_positive=float(np.mean(np.array(sel_test) > 0)),
                             winning_family=fams),
               all_val_sr_sd=float(np.std([q['val_sr'] for rows in allrows for q in rows])),
               all_test_sr_mean=float(np.mean([q['test_sr'] for rows in allrows for q in rows])))
    json.dump(out, open('results/exp5.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
