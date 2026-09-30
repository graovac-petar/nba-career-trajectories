"""Power of the tests for discrete structure: would they find career types if types existed?

Synthetic populations of the same size as the sample are drawn in the level-tilt plane (the first two principal
components of the resampled PER trajectory). Each population is a mixture of K equally frequent Gaussian classes whose
centres lie on a line, with adjacent centres separated by Delta within-class standard deviations (Mahalanobis
distance), and is then transformed linearly so that its total covariance equals that of the real coordinates.
Delta = 0 is a single Gaussian cloud. The same battery as for the real data is applied to every population:
(i) BIC choice among Gaussian mixtures with k = 1..6 and calibration of the BIC gain against 20 Gaussian-copula
references (detection if the gain exceeds every reference), (ii) BIC choice on normal scores, (iii) Hartigan's dip test
on the projection onto the first principal axis (p < 0.05), (iv) smallest average posterior probability of assignment.
Usage: python code/power_simulation.py
"""
import numpy as np, json, sys, warnings, time
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.linalg import sqrtm
import diptest

KMAX = 6; N_REF = 20; REPS = 20; N_INIT = 2
DELTAS = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.0, 1.5, 2.0, 2.5, 3.0, 4.0]
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/power_simulation.json'
zc = np.load('results/continuous_scores.npz'); S_real = zc['S'][:, :2]; n = len(S_real)
Sigma = np.cov(StandardScaler().fit_transform(S_real).T)

def fit_bic(Z, seed=0):
    Z = StandardScaler().fit_transform(Z); bics = []; models = []
    for k in range(1, KMAX + 1):
        g = GaussianMixture(k, covariance_type='full', n_init=N_INIT, random_state=seed, reg_covar=1e-5).fit(Z)
        bics.append(g.bic(Z)); models.append(g)
    kb = int(np.argmin(bics)) + 1; P = models[kb - 1].predict_proba(Z); lab = P.argmax(1)
    appa = min(P[lab == c, c].mean() for c in range(kb) if (lab == c).any()) if kb > 1 else 1.0
    return kb, bics[0] - bics[kb - 1], float(appa)

def copula(base, rng):
    U = (stats.rankdata(base, axis=0) - 0.5) / len(base); Zn = stats.norm.ppf(U)
    sim = rng.multivariate_normal(np.zeros(base.shape[1]), np.corrcoef(Zn.T), size=len(base))
    return np.column_stack([np.quantile(base[:, j], stats.norm.cdf(sim[:, j])) for j in range(base.shape[1])])

def simulate(K, delta, rng):
    lab = rng.randint(K, size=n); centres = (np.arange(K) - (K - 1) / 2) * delta
    W = rng.standard_normal((n, 2)); W[:, 0] += centres[lab]            # within-class covariance = identity
    u = np.array([1.0, 1.0]) / np.sqrt(2)                                  # classes differ in level and tilt
    R = np.array([[u[0], -u[1]], [u[1], u[0]]]); W = W @ R.T
    T = np.cov(W.T); A = np.real(sqrtm(Sigma)) @ np.linalg.inv(np.real(sqrtm(T)))   # match the real covariance
    return (W - W.mean(0)) @ A.T

rows = []; t0 = time.time()
for K in (3,):
    for delta in DELTAS:
        for r in range(REPS):
            rng = np.random.RandomState(1000 * K + 100 * int(10 * delta) + r)
            Y = simulate(K, delta, rng)
            kb, dbic, appa = fit_bic(Y)
            refs = [fit_bic(copula(Y, rng), seed=b)[1] for b in range(N_REF)]
            ns = stats.norm.ppf((stats.rankdata(Y, axis=0) - 0.5) / n); kns = fit_bic(ns)[0]
            dip_p = diptest.diptest(PCA(1).fit_transform(Y)[:, 0])[1]
            rows.append(dict(K=K, delta=delta, rep=r, k_bic=kb, dbic=dbic, detect_bic=bool(kb > 1 and dbic > max(refs)),
                             k_ns=kns, detect_ns=bool(kns > 1), dip_p=float(dip_p), detect_dip=bool(dip_p < 0.05), appa=appa,
                             detect_appa=bool(kb > 1 and appa >= 0.7)))
        sub = [x for x in rows if x['K'] == K and x['delta'] == delta]
        print(K, delta, {k: round(float(np.mean([x[k] for x in sub])), 2) for k in ['detect_bic', 'detect_ns', 'detect_dip', 'detect_appa', 'k_bic']},
              f'{time.time() - t0:.0f}s', flush=True)
summary = {}
for delta in sorted(set(x['delta'] for x in rows)):
    sub = [x for x in rows if x['delta'] == delta]
    summary[str(delta)] = {k: float(np.mean([x[k] for x in sub])) for k in ['detect_bic', 'detect_ns', 'detect_dip', 'detect_appa', 'k_bic']}
    summary[str(delta)]['detect_any'] = float(np.mean([x['detect_bic'] or x['detect_ns'] or x['detect_dip'] for x in sub]))
json.dump(dict(rows=rows, summary=summary, reps=REPS, n_ref=N_REF), open(OUT, 'w'), indent=1)
print(json.dumps(summary, indent=1))
