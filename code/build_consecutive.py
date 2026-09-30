"""Sensitivity: delete gaps and treat observed seasons as consecutive (the alignment used implicitly when gaps are ignored)."""
import numpy as np
d = np.load('results/trajectories_primary.npz'); X, M, pid = d['X'], d['M'], d['player_id']
Xc = np.zeros_like(X); Mc = np.zeros_like(M)
for i in range(len(X)):
    y = X[i][M[i]]; Xc[i, :len(y)] = y; Mc[i, :len(y)] = True
np.savez_compressed('results/trajectories_consecutive.npz', X=Xc, M=Mc, player_id=pid); print('ok', Mc.sum())
