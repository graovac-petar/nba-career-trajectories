"""Shared helpers: hand-crafted trajectory features and fixed-length resampling used by baselines and evaluation."""
import numpy as np, pandas as pd

def observed(X, M):
    return [X[i][M[i]] for i in range(len(X))]

def summary_features(X, M):
    """Nine interpretable per-player descriptors (used by the feature-based baselines)."""
    rows = []
    for i in range(len(X)):
        m = M[i]; y = X[i][m]; t = np.where(m)[0]; L = len(y)
        slope = np.polyfit(t, y, 1)[0] if L > 1 else 0.0
        rows.append(dict(mean=y.mean(), peak=y.max(), sd=y.std(), n_seasons=L, span=t.max() - t.min() + 1,
                         slope=slope, first=y[0], last=y[-1], peak_pos=(np.argmax(y) + 1) / L, frac_above_avg=(y >= 15).mean()))
    return pd.DataFrame(rows)

def resample(X, M, k=10):
    """Interpolate each observed trajectory onto k equally spaced points of normalised career time (0..1)
       and append log career length.  Gives a fixed-length vector for PCA / raw-UMAP baselines and for the
       shape-homogeneity criterion."""
    out = np.zeros((len(X), k + 1))
    grid = np.linspace(0, 1, k)
    for i in range(len(X)):
        m = M[i]; y = X[i][m]; t = np.where(m)[0].astype(float)
        u = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else np.zeros_like(t)
        out[i, :k] = np.interp(grid, u, y)
        out[i, k] = np.log(len(y))
    return out

def shape_r2(R, lab):
    """Share of total variance of the resampled-trajectory matrix explained by cluster membership (noise excluded)."""
    m = lab != -1
    Rm = R[m]; l = lab[m]
    sst = ((Rm - Rm.mean(0)) ** 2).sum()
    ssw = sum(((Rm[l == c] - Rm[l == c].mean(0)) ** 2).sum() for c in np.unique(l))
    return 1 - ssw / sst
