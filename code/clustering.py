"""
Step 3 - UMAP + HDBSCAN model selection with a prespecified criterion and held-out evaluation.

Development set = the autoencoder training players (80 %); held-out = the 20 % validation players.
Hyper-parameters are selected on the development set only; the selected model is then applied to the
held-out players with UMAP.transform + hdbscan.approximate_predict and evaluated there.
Finally the selected configuration is refitted on all players for interpretation.

Prespecified selection rule (fixed before running the grid):
   maximise DBCV (density-based cluster validity, Moulavi et al. 2014) on the development set
   subject to   noise fraction <= 0.25   and   number of clusters >= 3.
The 2025 composite score is recomputed only for comparison.
"""
import numpy as np, pandas as pd, json, itertools, warnings, argparse, os, time
import umap, hdbscan
from hdbscan.validity import validity_index
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
warnings.filterwarnings('ignore')

GRID = dict(n_components=[2, 3, 5, 10], n_neighbors=[10, 15, 20, 30, 50], min_dist=[0.0, 0.1],
            min_cluster_size=[15, 20, 25, 30, 40, 50], min_samples=[5, 10, 15])
CONSTRAINT = dict(max_noise=0.25, min_clusters=3)

def internal_metrics(E, lab):
    """Internal indices computed in the space that was clustered (UMAP space), noise excluded."""
    k = len(set(lab)) - (1 if -1 in lab else 0)
    noise = float((lab == -1).mean())
    out = dict(n_clusters=int(k), noise=noise)
    m = lab != -1
    if k >= 2 and m.sum() > k:
        out['silhouette'] = float(silhouette_score(E[m], lab[m]))
        out['calinski_harabasz'] = float(calinski_harabasz_score(E[m], lab[m]))
        out['davies_bouldin'] = float(davies_bouldin_score(E[m], lab[m]))
        try: out['dbcv'] = float(validity_index(E.astype(np.float64), lab))
        except Exception: out['dbcv'] = np.nan
    else:
        out.update(silhouette=np.nan, calinski_harabasz=np.nan, davies_bouldin=np.nan, dbcv=np.nan)
    return out

def composite_2025(m):
    """Composite used for selection in the 2025 submission (Silhouette 30 %, noise 30 %, CH 20 %, k 20 %)."""
    if np.isnan(m.get('silhouette', np.nan)): return np.nan
    sil = (m['silhouette'] + 1) / 2; ch = min(m['calinski_harabasz'] / 2000, 1.0); noise = 1 - m['noise']
    k = m['n_clusters']
    ks = 1.0 if 8 <= k <= 20 else 0.7 if 5 <= k < 8 else 0.8 if 20 < k <= 25 else 0.6 if 25 < k <= 35 else 0.3
    return 0.3 * sil + 0.3 * noise + 0.2 * ch + 0.2 * ks

def fit_umap(Zs, nc, nn, md, seed=0):
    return umap.UMAP(n_components=nc, n_neighbors=nn, min_dist=md, metric='euclidean', random_state=seed).fit(Zs)

def fit_hdb(E, mcs, ms):
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean',
                           cluster_selection_method='eom', prediction_data=True, gen_min_span_tree=True).fit(E)

def grid_search(Z, dev, tag, seed=0):
    sc = StandardScaler().fit(Z[dev]); Zs = sc.transform(Z)
    rows = []
    t0 = time.time()
    for nc, nn, md in itertools.product(GRID['n_components'], GRID['n_neighbors'], GRID['min_dist']):
        red = fit_umap(Zs[dev], nc, nn, md, seed)
        Edev = red.embedding_
        for mcs, ms in itertools.product(GRID['min_cluster_size'], GRID['min_samples']):
            h = fit_hdb(Edev, mcs, ms)
            m = internal_metrics(Edev, h.labels_)
            m.update(n_components=nc, n_neighbors=nn, min_dist=md, min_cluster_size=mcs, min_samples=ms,
                     composite_2025=composite_2025(m), dbcv_hdbscan=float(h.relative_validity_))
            rows.append(m)
        print(f'umap {nc},{nn},{md} done {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'results/grid_{tag}.csv', index=False)
    return df

def select(df, rule='dbcv'):
    ok = df[(df.noise <= CONSTRAINT['max_noise']) & (df.n_clusters >= CONSTRAINT['min_clusters'])]
    if rule == 'dbcv': return ok.sort_values('dbcv', ascending=False).iloc[0]
    if rule == 'silhouette': return ok.sort_values('silhouette', ascending=False).iloc[0]
    if rule == 'composite_2025': return df.sort_values('composite_2025', ascending=False).iloc[0]

def apply_config(Z, dev, cfg, seed=0):
    """Fit scaler+UMAP+HDBSCAN on dev, predict held-out, and also refit on all players."""
    sc = StandardScaler().fit(Z[dev]); Zs = sc.transform(Z)
    red = fit_umap(Zs[dev], int(cfg.n_components), int(cfg.n_neighbors), float(cfg.min_dist), seed)
    h = fit_hdb(red.embedding_, int(cfg.min_cluster_size), int(cfg.min_samples))
    Eho = red.transform(Zs[~dev])
    lab_ho, _ = hdbscan.approximate_predict(h, Eho)
    dev_m = internal_metrics(red.embedding_, h.labels_)
    ho_m = internal_metrics(Eho, lab_ho)
    # refit on all players
    sc2 = StandardScaler().fit(Z); Zs2 = sc2.transform(Z)
    red2 = fit_umap(Zs2, int(cfg.n_components), int(cfg.n_neighbors), float(cfg.min_dist), seed)
    h2 = fit_hdb(red2.embedding_, int(cfg.min_cluster_size), int(cfg.min_samples))
    all_m = internal_metrics(red2.embedding_, h2.labels_)
    return dict(dev=dev_m, heldout=ho_m, all=all_m, labels_all=h2.labels_, E_all=red2.embedding_,
                prob_all=h2.probabilities_, labels_dev=h.labels_, labels_ho=lab_ho, umap_all=red2, hdb_all=h2,
                strength_all=h2.probabilities_)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ae', default='primary_d64_s0'); a = ap.parse_args()
    d = np.load(f'results/ae_{a.ae}.npz'); Z, dev = d['Z'], d['train']
    df = grid_search(Z, dev, a.ae)
    out = {}
    for rule in ['dbcv', 'silhouette', 'composite_2025']:
        cfg = select(df, rule)
        r = apply_config(Z, dev, cfg)
        out[rule] = dict(config=cfg[['n_components', 'n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples']].to_dict(),
                         dev=r['dev'], heldout=r['heldout'], all=r['all'])
        np.savez_compressed(f'results/clusters_{a.ae}_{rule}.npz', labels=r['labels_all'], E=r['E_all'], prob=r['prob_all'],
                            labels_dev=r['labels_dev'], labels_ho=r['labels_ho'], player_id=d['player_id'], dev=dev)
        print(rule, json.dumps(out[rule], indent=1, default=float))
    json.dump(out, open(f'results/selection_{a.ae}.json', 'w'), indent=1, default=float)
