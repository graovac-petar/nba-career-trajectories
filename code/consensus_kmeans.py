"""
Step 4 (final) - Resampling-based consensus clustering (Monti et al. 2003; Fred & Jain 2005).

For a given representation (one or several embeddings of the same players, e.g. autoencoder seeds),
B ensemble members are generated: each member draws an 80 % subsample of players, optionally re-fits
UMAP on the subsample (new random state), and partitions it with k-means (k = 2..KMAX).  For each k the
co-association matrix C_k[i,j] (share of members containing i and j that placed them together) is
turned into a consensus partition by average-linkage agglomeration of 1 - C_k.

Reported per k:  PAC (share of pairs with 0.1 < C < 0.9; lower = less ambiguous, Senbabaoglu et al. 2014),
                 consensus silhouette (silhouette on 1 - C), mean within-cluster co-association.
Prespecified rule for k: the k that minimises PAC among k >= 3; ties broken by consensus silhouette.
Player-level stability = mean co-association with the other members of the player's consensus cluster.

Usage:  python code/consensus_kmeans.py --rep ae64 --space latent
"""
import numpy as np, pandas as pd, json, argparse, time, sys, warnings
sys.path.insert(0, 'code')
from features import summary_features, resample
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

KMAX = 12; SUBSAMPLE = 0.8

def load_representation(rep, spec='primary', seeds=range(5)):
    """Returns list of (name, matrix) with identical player order, and a 'mode' flag."""
    d = np.load(f'results/trajectories_{spec}.npz'); X, M = d['X'], d['M']
    if rep.startswith('ae'):
        dim = int(rep[2:]); out = []
        for s in seeds:
            try: out.append((f'ae{dim}_s{s}', np.load(f'results/ae_{spec}_d{dim}_s{s}.npz')['Z']))
            except FileNotFoundError: pass
        return out, 'vector'
    if rep == 'summary': return [('summary', StandardScaler().fit_transform(summary_features(X, M).values))], 'vector'
    if rep.startswith('summary_null') or rep.startswith('raw_null'):
        # Unimodal reference (Senbabaoglu et al. 2014): ONE simulated dataset per run, multivariate normal with the mean
        # and covariance of the standardised real features (same correlation structure, no cluster structure).
        base = summary_features(X, M).values if rep.startswith('summary') else resample(X, M)
        F = StandardScaler().fit_transform(base); b = int(rep[-1]); rng = np.random.RandomState(123 + b)
        return [(rep, rng.multivariate_normal(F.mean(0), np.cov(F.T), size=len(F)))], 'vector'
    if rep.startswith('summary_copula') or rep.startswith('raw_copula'):
        # Gaussian-copula reference: keeps every real marginal distribution (skewness, discreteness, bounds) and the
        # rank correlation structure, but imposes a unimodal (Gaussian) dependence structure with no clusters.
        from scipy import stats as st
        base = summary_features(X, M).values if rep.startswith('summary') else resample(X, M)
        b = int(rep[-1]); rng = np.random.RandomState(321 + b); n, p = base.shape
        U = (st.rankdata(base, axis=0) - 0.5) / n; Zn = st.norm.ppf(U)
        sim = rng.multivariate_normal(np.zeros(p), np.corrcoef(Zn.T), size=n)
        out = np.column_stack([np.quantile(base[:, j], st.norm.cdf(sim[:, j])) for j in range(p)])
        return [(rep, StandardScaler().fit_transform(out))], 'vector'
    if rep == 'shape':
        # Shape-only representation: resampled trajectory minus the player's own mean (removes level), no length
        Rr = resample(X, M)[:, :10]; return [('shape', StandardScaler().fit_transform(Rr - Rr.mean(1, keepdims=True)))], 'vector'
    if rep == 'raw': return [('raw', StandardScaler().fit_transform(resample(X, M)))], 'vector'
    if rep == 'pca': return [('pca', PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(resample(X, M))))], 'vector'
    if rep == 'dtw':
        seqs = [((X[i][M[i]] - 13.67) / 4.32).reshape(-1, 1) for i in range(len(X))]
        return [('dtw', seqs)], 'dtw'
    raise ValueError(rep)

def member_labels(S, idx, k_list, seed, mode, space, ward=False, clusterer='kmeans'):
    if mode == 'dtw':
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.utils import to_time_series_dataset
        ds = to_time_series_dataset([S[i] for i in idx])
        return {k: TimeSeriesKMeans(n_clusters=k, metric='dtw', metric_params={'global_constraint': 'sakoe_chiba', 'sakoe_chiba_radius': 3}, max_iter=10, n_init=1, random_state=seed, n_jobs=2).fit_predict(ds) for k in k_list}
    Ssub = S[idx]
    if space == 'umap5':
        import umap
        Ssub = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=seed).fit_transform(Ssub)
    if clusterer == 'hdbscan':
        import hdbscan
        lab = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=10).fit_predict(Ssub)
        return {k: lab for k in k_list}      # one density-based partition per member; noise (-1) never co-associates
    if ward: return {k: AgglomerativeClustering(k, linkage='ward').fit_predict(Ssub) for k in k_list}
    return {k: KMeans(k, n_init=5, random_state=seed).fit_predict(Ssub) for k in k_list}

def run(rep, space='latent', spec='primary', draws=8, ward=False, tag=None, kmax=KMAX, clusterer='kmeans', seeds=range(5), standardise=True):
    embs, mode = load_representation(rep, spec, seeds)
    tag = tag or f'{rep}_{space}{"_ward" if ward else ""}{"_hdbscan" if clusterer=="hdbscan" else ""}{"" if spec=="primary" else "_"+spec}'
    n = len(embs[0][1]); k_list = list(range(2, kmax + 1))
    both = np.zeros((n, n)); same = {k: np.zeros((n, n)) for k in k_list}
    rng = np.random.RandomState(0); r = 0; t0 = time.time()
    for ename, S in embs:
        if mode == 'vector' and standardise: S = StandardScaler().fit_transform(S)
        for dr in range(draws):
            idx = np.sort(rng.choice(n, int(SUBSAMPLE * n), replace=False))
            labs = member_labels(S, idx, k_list, r, mode, space, ward, clusterer); r += 1
            both[np.ix_(idx, idx)] += 1
            for k in k_list:
                lab = labs[k]
                for c in np.unique(lab):
                    if c == -1: continue
                    ii = idx[lab == c]; same[k][np.ix_(ii, ii)] += 1
        print(f'{tag}: {ename} done, {r} members, {time.time()-t0:.0f}s', flush=True)
    rows = []; parts = {}; iu = np.triu_indices(n, 1)
    for k in k_list:
        C = np.where(both > 0, same[k] / np.maximum(both, 1), 0); np.fill_diagonal(C, 1)
        D = 1 - C
        lab = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit(D).labels_
        sizes = np.bincount(lab)
        pac = float(((C[iu] > 0.1) & (C[iu] < 0.9)).mean())
        sil = float(silhouette_score(D, lab, metric='precomputed'))
        within = float(np.mean([C[np.ix_(lab == c, lab == c)][np.triu_indices(sizes[c], 1)].mean() for c in range(k) if sizes[c] > 1]))
        stab = np.array([(C[i, lab == lab[i]].sum() - 1) / max(sizes[lab[i]] - 1, 1) for i in range(n)])
        rows.append(dict(rep=tag, k=k, members=r, pac=pac, silhouette_consensus=sil, mean_within_coassoc=within,
                         min_size=int(sizes.min()), max_size=int(sizes.max()), median_stability=float(np.median(stab)),
                         frac_stable=float((stab >= 0.5).mean())))
        parts[k] = (lab, stab, C if k in (8,) else None)
    df = pd.DataFrame(rows)
    # Selection rule: maximise the consensus Silhouette among k >= 3 (PAC is reported as a diagnostic; it decreases
    # monotonically in k for every representation and therefore cannot select k on its own)
    ok = df[df.k >= 3]; kbest = int(ok.sort_values(['silhouette_consensus', 'pac'], ascending=[False, True]).iloc[0].k)
    df['selected'] = df.k == kbest
    df.to_csv(f'results/consensus_{tag}.csv', index=False)
    lab, stab, _ = parts[kbest]
    Cb = np.where(both > 0, same[kbest] / np.maximum(both, 1), 0); np.fill_diagonal(Cb, 1)
    np.savez_compressed(f'results/consensus_{tag}.npz', labels=lab, stability=stab, k=kbest, C=Cb.astype(np.float16),
                        labels8=parts[8][0], stability8=parts[8][1], C8=parts[8][2].astype(np.float16),
                        labels_by_k=np.array([parts[k][0] for k in k_list]), stability_by_k=np.array([parts[k][1] for k in k_list]), k_list=np.array(k_list))
    print(df.to_string(), '\nselected k =', kbest, flush=True)
    return df, kbest

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rep', default='ae64'); ap.add_argument('--space', default='latent')
    ap.add_argument('--spec', default='primary'); ap.add_argument('--draws', type=int, default=8); ap.add_argument('--ward', action='store_true'); ap.add_argument('--clusterer', default='kmeans'); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--tag', default=None); ap.add_argument('--nostd', action='store_true')
    a = ap.parse_args()
    run(a.rep, a.space, a.spec, a.draws, a.ward, tag=a.tag, clusterer=a.clusterer, seeds=[int(x) for x in a.seeds.split(',')], standardise=not a.nostd)
