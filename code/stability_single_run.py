"""Reproducibility of a single UMAP+HDBSCAN run (the 2025 design) across random seeds and autoencoder seeds."""
import numpy as np, pandas as pd, itertools, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from clustering import fit_umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ari
CONFIGS = [(10, 10, 0.0, 25, 15), (5, 15, 0.0, 30, 10), (5, 20, 0.0, 22, 7), (2, 15, 0.0, 40, 5)]
rows = []; labs_all = {}
N_AE_SEEDS = 3; N_UMAP_SEEDS = 10
for s in range(N_AE_SEEDS):
    Z = np.load(f'results/ae_primary_d64_s{s}.npz')['Z']
    Zs = StandardScaler().fit_transform(Z)
    for cfg in CONFIGS:
        Es = [fit_umap(Zs, cfg[0], cfg[1], cfg[2], u).embedding_ for u in range(N_UMAP_SEEDS)]
        for method in ['eom', 'leaf']:
            labs = []
            for u, E in enumerate(Es):
                lab = hdbscan.HDBSCAN(min_cluster_size=cfg[3], min_samples=cfg[4], cluster_selection_method=method).fit_predict(E)
                labs.append(lab); labs_all[(s, cfg, method, u)] = lab
            ks = [int(l.max() + 1) for l in labs]; noise = [float((l == -1).mean()) for l in labs]
            big = [float(np.bincount(l[l >= 0]).max() / len(l)) for l in labs]
            pair = [ari(a, b) for a, b in itertools.combinations(labs, 2)]
            rows.append(dict(ae_seed=s, config=str(cfg), method=method, k_min=min(ks), k_median=float(np.median(ks)), k_max=max(ks),
                             noise_mean=np.mean(noise), largest_cluster_share=np.mean(big), ari_mean=np.mean(pair), ari_min=np.min(pair), ari_max=np.max(pair)))
        print('seed', s, cfg, flush=True)
df = pd.DataFrame(rows); df.to_csv('results/stability_single_run.csv', index=False); print(df.round(3).to_string())
# across autoencoder seeds (same UMAP seed 0)
rows2 = []
for cfg in CONFIGS:
    for method in ['eom', 'leaf']:
        L = [labs_all[(s, cfg, method, 0)] for s in range(N_AE_SEEDS)]
        pair = [ari(a, b) for a, b in itertools.combinations(L, 2)]
        rows2.append(dict(config=str(cfg), method=method, ari_mean_across_ae_seeds=np.mean(pair), ari_min=np.min(pair)))
pd.DataFrame(rows2).to_csv('results/stability_across_ae_seeds.csv', index=False); print(pd.DataFrame(rows2).round(3).to_string())
