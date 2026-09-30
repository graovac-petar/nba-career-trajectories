"""
Step 5 - Common evaluation of consensus partitions: internal (own-space silhouette), shape homogeneity,
reproducibility (PAC, consensus silhouette), and external validity against variables NOT used in clustering
(position, draft status, All-Star / All-NBA selection, Win Shares, Box Plus-Minus).
Usage: python code/evaluate.py  (evaluates every results/consensus_*.npz)
"""
import numpy as np, pandas as pd, glob, sys, json, warnings
sys.path.insert(0, 'code')
from features import resample, summary_features, shape_r2
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
warnings.filterwarnings('ignore')

def cramers_v(x, y):
    ct = pd.crosstab(x, y); chi2 = stats.chi2_contingency(ct)[0]; n = ct.values.sum()
    r, c = ct.shape; return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))), float(stats.chi2_contingency(ct)[1])

def kw_epsilon2(groups):
    """Kruskal-Wallis H and epsilon-squared effect size (Tomczak & Tomczak 2014)."""
    H, p = stats.kruskal(*groups); n = sum(len(g) for g in groups)
    return float(H), float(p), float((H - len(groups) + 1) / (n - len(groups)))

def load_external():
    m = pd.read_csv('results/players_primary.csv')
    dm = pd.read_csv('results/draft_match.csv'); m = m.merge(dm, on='player_id', how='left')
    m['draft_cat'] = np.select([m['pick'] <= 14, m['pick'] <= 30, m['pick'] > 30], ['lottery', 'first_round', 'second_round'], 'undrafted')
    m['all_star_ever'] = m['all_star_n'] > 0; m['all_nba_ever'] = m['all_nba_n'] > 0
    return m

def evaluate_partition(lab, m, R):
    out = dict(k=int(lab.max() + 1), shape_r2=shape_r2(R, lab))
    for var in ['pos_mode', 'draft_cat', 'all_star_ever', 'all_nba_ever']:
        v, p = cramers_v(lab, m[var]); out[f'cramersV_{var}'] = v; out[f'p_{var}'] = p; out[f'ami_{var}'] = float(ami(m[var].astype(str), lab))
    for var in ['sum_ws', 'mean_bpm', 'career_games', 'mean_per', 'n_valid']:
        H, p, e2 = kw_epsilon2([m[var].values[lab == c] for c in np.unique(lab)]); out[f'eps2_{var}'] = e2; out[f'p_{var}'] = p
    return out

if __name__ == '__main__':
    m = load_external(); d = np.load('results/trajectories_primary.npz'); R = resample(d['X'], d['M'])
    rows = []
    for f in sorted(glob.glob('results/consensus_*.npz')):
        tag = f.split('consensus_')[1][:-4]
        z = np.load(f); lab = z['labels']
        if len(lab) != len(m): continue
        r = dict(rep=tag); r.update(evaluate_partition(lab, m, R))
        cs = pd.read_csv(f'results/consensus_{tag}.csv'); sel = cs[cs.selected].iloc[0]
        r.update(pac=sel.pac, silhouette_consensus=sel.silhouette_consensus, median_stability=sel.median_stability, frac_stable=sel.frac_stable)
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv('results/evaluation_table.csv', index=False)
    print(df.round(3).to_string())
