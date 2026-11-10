"""Robustness to the performance metric: build BPM trajectories for the same players / seasons as the primary PER specification."""
import numpy as np, pandas as pd
ps = pd.read_csv('results/player_seasons.csv'); m = pd.read_csv('results/players_primary.csv')
d = np.load('results/trajectories_primary.npz'); X, M, pid = d['X'], d['M'], d['player_id']
Xb = np.zeros_like(X)
look = ps.set_index(['player_id', 'season_end'])['bpm']
for i, p in enumerate(pid):
    y0 = int(m.loc[m.player_id == p, 'debut'].iloc[0])
    for t in np.where(M[i])[0]:
        Xb[i, t] = look.get((p, y0 + t), np.nan)
Mb = M & ~np.isnan(Xb); Xb = np.nan_to_num(Xb)
np.savez_compressed('results/trajectories_primary_bpm.npz', X=Xb, M=Mb, player_id=pid)
print('players', len(pid), 'valid BPM obs', Mb.sum(), 'of', M.sum())
