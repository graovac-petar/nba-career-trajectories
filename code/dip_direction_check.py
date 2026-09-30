"""Power of the dip test in the simulation of power_simulation.py when the projection axis is not the first
principal component: for each simulated sample the dip test is applied along 36 directions of the whitened
plane and the smallest p-value is compared with 0.05/36 (Bonferroni). This bounds what a better choice of
axis could achieve.  Usage: python code/dip_direction_check.py"""
import numpy as np, json, sys
sys.argv = [sys.argv[0]]
src = open('code/power_simulation.py').read()
exec(src.split('rows = []')[0])
import diptest
from sklearn.decomposition import PCA
angles = np.linspace(0, np.pi, 36, endpoint=False)
out = {}
for delta in [0, 1.5, 2, 2.5, 3, 4]:
    pc, best = [], []
    for r in range(20):
        rng = np.random.RandomState(1000 * 3 + 100 * int(10 * delta) + r)
        Y = simulate(3, delta, rng)
        pc.append(diptest.diptest(PCA(1).fit_transform(Y)[:, 0])[1] < 0.05)
        Wt = Y @ np.linalg.inv(np.real(sqrtm(np.cov(Y.T))))
        best.append(min(diptest.diptest(Wt @ np.array([np.cos(a), np.sin(a)]))[1] for a in angles) < 0.05 / len(angles))
    out[str(delta)] = dict(dip_pc1=float(np.mean(pc)), dip_best_direction=float(np.mean(best)))
    print(delta, out[str(delta)])
json.dump(out, open('results/dip_direction.json', 'w'), indent=1)
