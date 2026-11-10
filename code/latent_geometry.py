"""Why do autoencoder partitions depend on the training seed?

For the five d = 64 autoencoders, compare the latent geometries directly: correlation of the between-player distance
matrices (Pearson and Spearman on all pairs), overlap of the ten nearest neighbours of each player, and the variance
of one seed's latent codes explained by a linear map from another seed (cross-validated R^2). For reference the same
quantities are computed between the latent space and the summary-statistic and resampled-trajectory spaces, and the
agreement of single k-means runs (k = 4) on each latent space is reported. High distance agreement combined with low
partition agreement would indicate that k-means cuts a continuum differently; low distance agreement indicates that
the learned geometry itself differs between seeds.
Usage: python code/latent_geometry.py
"""
import numpy as np, json, sys, itertools, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import summary_features, resample
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

d = np.load('results/trajectories_primary.npz'); X, M = d['X'], d['M']
Z = {s: StandardScaler().fit_transform(np.load(f'results/ae_primary_d64_s{s}.npz')['Z']) for s in range(5)}
ref = {'summary': StandardScaler().fit_transform(summary_features(X, M).values),
       'resampled': StandardScaler().fit_transform(resample(X, M))}
rng = np.random.RandomState(0); n = len(X)
pairs_idx = None

def dist(A): return pdist(A)
def knn_sets(A, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(A); return [set(r[1:]) for r in nn.kneighbors(A, return_distance=False)]
def knn_overlap(A, B, k=10):
    a, b = knn_sets(A, k), knn_sets(B, k); return float(np.mean([len(x & y) / k for x, y in zip(a, b)]))
def lin_r2(A, B):
    return float(np.mean(cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 13)), A, B, cv=5, scoring='r2')))

D = {s: dist(Z[s]) for s in Z}; Dr = {k: dist(v) for k, v in ref.items()}
sub = rng.choice(len(D[0]), 200000, replace=False)        # Spearman on a random subset of pairs
rows = []
for a, b in itertools.combinations(range(5), 2):
    rows.append(dict(a=a, b=b, pearson=float(np.corrcoef(D[a], D[b])[0, 1]), spearman=float(spearmanr(D[a][sub], D[b][sub])[0]),
                     knn10=knn_overlap(Z[a], Z[b]), linear_r2=lin_r2(Z[a], Z[b])))
K_CUT = 4                                            # the number of groups selected for the summary statistics
km = {s: KMeans(K_CUT, n_init=10, random_state=0).fit_predict(Z[s]) for s in Z}
km_ari = [float(ari(km[a], km[b])) for a, b in itertools.combinations(range(5), 2)]
ref_rows = {k: dict(pearson=float(np.mean([np.corrcoef(D[s], Dr[k])[0, 1] for s in Z])), knn10=float(np.mean([knn_overlap(Z[s], ref[k]) for s in Z])))
            for k in ref}
# k-means on the same latent space with different initialisations only: how much of the disagreement is the cut itself?
km_init = [KMeans(K_CUT, n_init=1, random_state=r).fit_predict(Z[0]) for r in range(10)]
init_ari = [float(ari(x, y)) for x, y in itertools.combinations(km_init, 2)]
# knn overlap expected by chance
chance = 10 / (n - 1)
# effective dimensionality (inverse participation ratio of the PCA spectrum): how many directions of similar variance
from sklearn.decomposition import PCA
def eff_dim(A):
    ev = PCA().fit(A).explained_variance_ratio_; return float(1 / np.sum(ev ** 2))
effdim = dict(latent=float(np.mean([eff_dim(Z[s]) for s in Z])), **{k: eff_dim(v) for k, v in ref.items()})
out = dict(seed_pairs=rows, mean=dict(pearson=float(np.mean([r['pearson'] for r in rows])), spearman=float(np.mean([r['spearman'] for r in rows])),
                                         knn10=float(np.mean([r['knn10'] for r in rows])), linear_r2=float(np.mean([r['linear_r2'] for r in rows]))),
           range=dict(pearson=[min(r['pearson'] for r in rows), max(r['pearson'] for r in rows)], knn10=[min(r['knn10'] for r in rows), max(r['knn10'] for r in rows)]),
           kmeans8_ari_between_seeds=dict(mean=float(np.mean(km_ari)), min=min(km_ari), max=max(km_ari)),
           kmeans8_ari_between_inits_seed0=dict(mean=float(np.mean(init_ari)), min=min(init_ari), max=max(init_ari)),
           vs_reference=ref_rows, knn_chance=chance, effective_dimension=effdim)
json.dump(out, open('results/latent_geometry.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'seed_pairs'}, indent=1))
