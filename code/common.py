"""Shared helpers for the extended experiments (5-8)."""
import itertools
import numpy as np
from scipy.stats import norm

GAMMA = 0.5772156649


def z_fst(K):
    """Bracket term of the False Strategy Theorem for K trials (0 for K <= 1)."""
    K = np.asarray(K, dtype=float)
    out = (1 - GAMMA) * norm.ppf(1 - 1 / np.maximum(K, 1.0000001)) + \
        GAMMA * norm.ppf(1 - 1 / (np.maximum(K, 1.0000001) * np.e))
    return np.where(K <= 1, 0.0, out)


def ann_sr(r, axis=0):
    return r.mean(axis) / r.std(axis, ddof=1) * np.sqrt(252)


def psr(sr, sr0, T, g3=0.0, g4=3.0):
    """Probabilistic Sharpe ratio, per-period SRs (Bailey & Lopez de Prado, 2012)."""
    return norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))


# ---------- moving-average crossover rule grid ----------
FAST = list(range(2, 51, 4))
SLOW = list(range(20, 201, 15))
PAIRS = [(f, s) for f in FAST for s in SLOW if f < s]


def ma_positions(p):
    """Positions (+1/-1) of the 156 crossover rules for log-price path p, aligned so that
    position[t] is held over return p[t+1]-p[t]. Returns (positions, returns)."""
    cs = np.concatenate([[0], np.cumsum(p)])
    def ma(n):
        m = np.full(len(p), np.nan)
        m[n - 1:] = (cs[n:] - cs[:-n]) / n
        return m
    cache = {n: ma(n) for n in set(FAST) | set(SLOW)}
    ret = np.diff(p)
    P = np.empty((len(ret), len(PAIRS)))
    for k, (f, s) in enumerate(PAIRS):
        P[:, k] = np.nan_to_num(np.sign(cache[f] - cache[s])[:-1])
    return P, ret


def rule_returns(P, ret, cost_bps=0.0):
    R = P * ret[:, None]
    if cost_bps:
        turnover = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0))
        R = R - turnover * cost_bps / 1e4
    return R


def cscv_pbo(R, S=16):
    """Probability of backtest overfitting by CSCV (Bailey et al., 2017)."""
    n = (R.shape[0] // S) * S
    R = R[:n]
    blocks = np.array_split(np.arange(n), S)
    lam = []
    for comb in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.setdiff1d(np.arange(n), tr)
        s_tr = R[tr].mean(0) / R[tr].std(0)
        s_te = R[te].mean(0) / R[te].std(0)
        j = np.argmax(s_tr)
        w = (s_te < s_te[j]).mean() + 0.5 / len(s_te)
        w = min(max(w, 1e-6), 1 - 1e-6)
        lam.append(np.log(w / (1 - w)))
    lam = np.array(lam)
    return float((lam <= 0).mean())


# ---------- effective number of trials ----------
def k_participation(C):
    """Participation ratio of the correlation matrix eigenvalues."""
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def k_liji(C):
    """Li & Ji (2005) effective number of independent tests."""
    lam = np.abs(np.linalg.eigvalsh(C))
    return float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))


def k_cluster(C, kmax=150):
    """Hierarchical clustering on correlation distance; number of clusters by best silhouette
    (a simplified version of the clustering approach of Lopez de Prado & Lewis, 2019)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method='average')
    n = C.shape[0]
    grid = sorted({k for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150] if k < min(n, kmax + 1)})
    best, bestk = -2, 1
    for k in grid:
        lab = fcluster(Z, k, criterion='maxclust')
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(D, lab, metric='precomputed')
        if s > best:
            best, bestk = s, len(set(lab))
    return float(bestk)
