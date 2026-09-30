"""Tests for discrete structure versus a continuum: gap statistic (Tibshirani et al. 2001) with a PCA-aligned uniform
reference and a multivariate-normal reference, and Hartigan's dip test of unimodality on the leading principal components."""
import numpy as np, pandas as pd, json, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import summary_features, resample
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import diptest
d = np.load('results/trajectories_primary.npz'); X, M = d['X'], d['M']
out = {}
def wk(S, k, seed=0):
    km = KMeans(k, n_init=5, random_state=seed).fit(S); return km.inertia_
def gap(S, name, B=20, kmax=12):
    rng = np.random.RandomState(0); pca = PCA().fit(S); P = pca.transform(S)
    lo, hi = P.min(0), P.max(0); mu, cov = S.mean(0), np.cov(S.T)
    rows = []
    for k in range(1, kmax + 1):
        w = np.log(wk(S, k))
        ref_u = [np.log(wk(pca.inverse_transform(rng.uniform(lo, hi, size=P.shape)), k, b)) for b in range(B)]
        ref_n = [np.log(wk(rng.multivariate_normal(mu, cov, size=len(S)), k, b)) for b in range(B)]
        rows.append(dict(rep=name, k=k, logW=w, gap_uniform=np.mean(ref_u) - w, sd_uniform=np.std(ref_u) * np.sqrt(1 + 1 / B), gap_normal=np.mean(ref_n) - w, sd_normal=np.std(ref_n) * np.sqrt(1 + 1 / B)))
        print(name, k, rows[-1]['gap_uniform'].round(3), rows[-1]['gap_normal'].round(3), flush=True)
    return pd.DataFrame(rows)
reps = {'summary': StandardScaler().fit_transform(summary_features(X, M).values), 'raw': StandardScaler().fit_transform(resample(X, M)),
        'ae64_s0': StandardScaler().fit_transform(np.load('results/ae_primary_d64_s0.npz')['Z'])}
G = pd.concat([gap(S, n) for n, S in reps.items()]); G.to_csv('results/gap_statistic.csv', index=False)
# gap rule of Tibshirani: smallest k with gap(k) >= gap(k+1) - s(k+1)
for n in reps:
    g = G[G.rep == n].reset_index(drop=True)
    for ref in ['uniform', 'normal']:
        ks = [int(g.k[i]) for i in range(len(g) - 1) if g[f'gap_{ref}'][i] >= g[f'gap_{ref}'][i + 1] - g[f'sd_{ref}'][i + 1]]
        out[f'gap_k_{n}_{ref}'] = ks[0] if ks else None; out[f'gap_max_{n}_{ref}'] = float(g[f'gap_{ref}'].max()); out[f'gap_argmax_{n}_{ref}'] = int(g.k[g[f'gap_{ref}'].idxmax()])
# dip test on leading PCs and on single features
for n, S in reps.items():
    P = PCA(3).fit_transform(S)
    for j in range(3):
        dip, p = diptest.diptest(P[:, j]); out[f'dip_{n}_pc{j+1}'] = float(dip); out[f'dip_p_{n}_pc{j+1}'] = float(p)
F = summary_features(X, M)
for c in ['mean', 'n_seasons', 'peak', 'slope']:
    dip, p = diptest.diptest(F[c].values); out[f'dip_feat_{c}'] = float(dip); out[f'dip_p_feat_{c}'] = float(p)
json.dump(out, open('results/structure_tests.json', 'w'), indent=1); print(json.dumps(out, indent=1))
