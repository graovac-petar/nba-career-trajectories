"""What the mixture components of the summary statistics capture: the BIC-selected mixtures of mixture_tests.py are
refitted (same settings and seeds) and each component is checked for degenerate content, i.e. a feature that is
constant within the component (e.g. share of seasons at or above PER 15 exactly zero, or exactly two seasons).
Usage: python code/mixture_diagnostics.py  (after mixture_tests.py)"""
import numpy as np, pandas as pd, json, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import summary_features
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
mt = json.load(open('results/mixture_tests.json'))
d = np.load('results/trajectories_primary.npz'); F = summary_features(d['X'], d['M'])
meta = pd.read_csv('results/players_primary.csv')
out = {}
for key, sub in [('summary', F), ('summary5', F[meta.n_valid.values >= 5].reset_index(drop=True))]:
    k = mt[key]['real']['k_bic']; Z = StandardScaler().fit_transform(sub.values)
    lab = GaussianMixture(k, covariance_type='full', n_init=5, random_state=0, reg_covar=1e-5).fit(Z).predict(Z)
    comps = []
    for c in np.unique(lab):
        g = sub[lab == c]
        comps.append(dict(n=int(len(g)), frac0=float((g.frac_above_avg == 0).mean()), two=float((g.n_seasons == 2).mean()),
                          const=[col for col in sub.columns if g[col].nunique() == 1]))
    out[key] = dict(k=k, components=comps,
                    n_pure_frac0=int(sum(c['n'] for c in comps if c['frac0'] >= 0.99)),
                    n_pure_two=int(sum(c['n'] for c in comps if c['two'] >= 0.99)),
                    n_in_degenerate=int(sum(c['n'] for c in comps if c['const'])),
                    share_frac0_sample=float((sub.frac_above_avg == 0).mean()))
    print(key, json.dumps({k_: v for k_, v in out[key].items() if k_ != 'components'}), [(c['n'], c['const']) for c in comps])
json.dump(out, open('results/mixture_diagnostics.json', 'w'), indent=1)
