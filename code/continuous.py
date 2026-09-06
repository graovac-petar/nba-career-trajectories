"""A continuous description of the career continuum: principal components of the resampled PER trajectory (PER units),
their relation to level, length, slope and peak timing, and tests for discrete structure in shape once level is removed."""
import numpy as np, pandas as pd, json, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import resample, summary_features
from sklearn.decomposition import PCA
from scipy import stats
import diptest
d = np.load('results/trajectories_primary.npz'); X, M = d['X'], d['M']
meta = pd.read_csv('results/players_primary.csv'); F = summary_features(X, M)
R = resample(X, M)[:, :10]                       # 10 points of normalised career time, PER units
pca = PCA().fit(R); S = pca.transform(R)
out = dict(var_ratio=pca.explained_variance_ratio_[:5].tolist(), loadings=pca.components_[:3].tolist(), mean_curve=pca.mean_.tolist())
for j in range(3):
    out[f'pc{j+1}_corr'] = {v: float(np.corrcoef(S[:, j], F[v])[0, 1]) for v in ['mean', 'n_seasons', 'slope', 'peak_pos', 'peak', 'sd']}
    out[f'pc{j+1}_corr_ws'] = float(np.corrcoef(S[:, j], meta.sum_ws)[0, 1])
# shape-only: remove level
Rs = R - R.mean(1, keepdims=True); pcs = PCA().fit(Rs); Ss = pcs.transform(Rs)
out['shape_var_ratio'] = pcs.explained_variance_ratio_[:5].tolist(); out['shape_loadings'] = pcs.components_[:2].tolist()
for j in range(2):
    dip, p = diptest.diptest(Ss[:, j]); out[f'shape_dip_pc{j+1}'] = float(dip); out[f'shape_dip_p_pc{j+1}'] = float(p)
dip, p = diptest.diptest(F['peak_pos'].values); out['dip_peak_pos'] = float(dip); out['dip_p_peak_pos'] = float(p)
dip, p = diptest.diptest(F['slope'].values); out['dip_slope'] = float(dip); out['dip_p_slope'] = float(p)
# distribution of peak timing (Wakim & Jin: early / middle / late peakers)
pp = F['peak_pos'].values; out['peak_pos_quantiles'] = np.quantile(pp, [.1, .25, .5, .75, .9]).tolist()
out['frac_peak_first_third'] = float((pp <= 1/3).mean()); out['frac_peak_last_third'] = float((pp > 2/3).mean())
np.savez_compressed('results/continuous_scores.npz', S=S[:, :3], Ss=Ss[:, :2], player_id=d['player_id'])
json.dump(out, open('results/continuous.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if 'loadings' not in k and 'curve' not in k}, indent=1))
