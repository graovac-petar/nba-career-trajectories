"""Direct test of discrete classes against a continuous distribution with finite Gaussian mixtures.

For two representations (the three continuous coordinates and the ten summary statistics), Gaussian mixtures with
k = 1..KMAX full-covariance components are fitted and compared by BIC (Fraley & Raftery 2002). Because a Gaussian
mixture also improves on a single Gaussian when the data are unimodal but non-Gaussian (skewed, bounded or discrete
marginals), the observed BIC improvement and class separation are calibrated against Gaussian-copula references
that keep every real marginal and the rank correlations but have no class structure. Class separation is measured
by the relative entropy of the posterior classification (Celeux & Soromenho 1996) and by the average posterior
probability of assignment (Nagin 2005). On marginally normalised data (normal scores) a parametric bootstrap of the
likelihood-ratio statistic for one against two components is also computed (McLachlan 1987).
Usage: python code/mixture_tests.py
"""
import numpy as np, pandas as pd, json, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import summary_features
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats

KMAX = 10; N_NULL = 20; B_LRT = 99; N_INIT = 5
d = np.load('results/trajectories_primary.npz'); X, M = d['X'], d['M']
meta = pd.read_csv('results/players_primary.csv')
zc = np.load('results/continuous_scores.npz')
F = summary_features(X, M).values; long_ = meta.n_valid.values >= 5
# Careers with two or three seasons put the summary statistics on low-dimensional lattices (e.g. with two seasons
# the peak, SD, first and last values are functions of two numbers), which a full-covariance mixture captures with
# near-degenerate components; the summary-statistic test is therefore also run on careers with at least five seasons.
from features import resample
from sklearn.decomposition import PCA
db = np.load('results/trajectories_primary_bpm.npz'); assert (db['player_id'] == d['player_id']).all()
Sb = PCA(2).fit_transform(resample(db['X'], db['M'])[:, :10])       # level and tilt of the BPM trajectory
reps = {'bpm_pc12': Sb, 'bpm_coords': np.column_stack([Sb, np.log(meta.n_valid.values)]),
        'coords': np.column_stack([zc['S'][:, :2], np.log(meta.n_valid.values)]),
        'summary': F, 'summary5': F[long_], 'coords5': np.column_stack([zc['S'][:, :2], np.log(meta.n_valid.values)])[long_], 'pc12': zc['S'][:, :2]}

def fit_all(Z, seed=0):
    Z = StandardScaler().fit_transform(Z); out = []
    for k in range(1, KMAX + 1):
        g = GaussianMixture(k, covariance_type='full', n_init=N_INIT, random_state=seed, reg_covar=1e-5).fit(Z)
        out.append((k, g.bic(Z), g.score(Z) * len(Z), g))
    kb = min(out, key=lambda r: r[1])[0]; g = out[kb - 1][3]
    P = g.predict_proba(Z); lab = P.argmax(1)
    if kb > 1:
        ent = -(P * np.log(np.clip(P, 1e-300, 1))).sum() / (len(Z) * np.log(kb)); rel_entropy = 1 - ent
        appa = [P[lab == c, c].mean() for c in range(kb) if (lab == c).any()]
    else:
        rel_entropy = np.nan; appa = [1.0]
    return dict(k_bic=kb, bic=[r[1] for r in out], dbic=out[0][1] - out[kb - 1][1], rel_entropy=rel_entropy,
                appa_min=float(min(appa)), appa_mean=float(np.mean(appa)), loglik=[r[2] for r in out])

def copula_sample(base, rng):
    n, p = base.shape
    U = (stats.rankdata(base, axis=0) - 0.5) / n; Zn = stats.norm.ppf(U)
    sim = rng.multivariate_normal(np.zeros(p), np.corrcoef(Zn.T), size=n)
    return np.column_stack([np.quantile(base[:, j], stats.norm.cdf(sim[:, j])) for j in range(p)])

def normal_scores(Z):
    return stats.norm.ppf((stats.rankdata(Z, axis=0) - 0.5) / len(Z))

def boot_lrt(Z, B=B_LRT, seed=0):
    """Parametric bootstrap of -2 log LR for 1 vs 2 components (McLachlan 1987)."""
    Z = StandardScaler().fit_transform(Z)
    def lr(Y):
        g1 = GaussianMixture(1, covariance_type='full').fit(Y); g2 = GaussianMixture(2, covariance_type='full', n_init=N_INIT, random_state=0).fit(Y)
        return 2 * len(Y) * (g2.score(Y) - g1.score(Y))
    obs = lr(Z); rng = np.random.RandomState(seed); mu, cov = Z.mean(0), np.cov(Z.T)
    null = [lr(rng.multivariate_normal(mu, cov, size=len(Z))) for _ in range(B)]
    return float(obs), float((1 + sum(n_ >= obs for n_ in null)) / (B + 1)), float(np.percentile(null, 95))

out = {}
for name, Z in reps.items():
    real = fit_all(Z); rng = np.random.RandomState(2026)
    nulls = [fit_all(copula_sample(Z, rng), seed=b) for b in range(N_NULL)]
    ns = normal_scores(Z); real_ns = fit_all(ns)
    lrt_obs, lrt_p, lrt_crit = boot_lrt(ns)
    rec = dict(real=real, null_k=[n_['k_bic'] for n_ in nulls], null_dbic=[n_['dbic'] for n_ in nulls],
               null_entropy=[n_['rel_entropy'] for n_ in nulls], null_appa_min=[n_['appa_min'] for n_ in nulls],
               ns=real_ns, lrt_ns=dict(obs=lrt_obs, p=lrt_p, crit95=lrt_crit))
    rec['p_dbic'] = float((1 + sum(v >= real['dbic'] for v in rec['null_dbic'])) / (N_NULL + 1))
    ent_null = [v for v in rec['null_entropy'] if not np.isnan(v)]
    rec['p_entropy'] = float((1 + sum(v >= real['rel_entropy'] for v in ent_null)) / (len(ent_null) + 1)) if ent_null and not np.isnan(real['rel_entropy']) else None
    out[name] = rec
    print(name, 'k_BIC', real['k_bic'], 'dBIC', round(real['dbic'], 1), 'entropy', round(real['rel_entropy'], 3), 'APPA min', round(real['appa_min'], 3),
          '| null k', rec['null_k'], 'null dBIC median', round(float(np.median(rec['null_dbic'])), 1), 'p', rec['p_dbic'],
          '| null entropy median', round(float(np.nanmedian(rec['null_entropy'])), 3), 'p', rec['p_entropy'],
          '| normal scores k', real_ns['k_bic'], 'entropy', round(real_ns['rel_entropy'], 3) if not np.isnan(real_ns['rel_entropy']) else None,
          'LRT', round(lrt_obs, 1), 'p', lrt_p, flush=True)
for r in out.values():
    for key in ['real', 'ns']:
        r[key] = {k: v for k, v in r[key].items() if k != 'loglik'}
json.dump(out, open('results/mixture_tests.json', 'w'), indent=1, default=float)
