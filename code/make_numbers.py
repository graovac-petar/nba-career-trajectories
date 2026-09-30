"""Step 7 - generate paper/numbers.tex (every numeric value quoted in the manuscript) and the LaTeX tables
from the result files, so that text and results cannot drift apart."""
import numpy as np, pandas as pd, json, glob, os, sys, warnings
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import summary_features, resample, shape_r2
from sklearn.metrics import adjusted_rand_score as ari
from scipy import stats
os.makedirs('paper/tables', exist_ok=True)
N = {}
def num(k, v, fmt='{:.2f}'):
    t = fmt.format(v) if not isinstance(v, str) else v
    N[k] = '\\ensuremath{-}' + t[1:] if t.startswith('-') and len(t) > 1 and t[1].isdigit() else t
def fmtn(x): return f'{int(x):,}'.replace(',', ',')

# ---------------- data ----------------
ps = pd.read_csv('results/player_seasons.csv'); raw = pd.read_csv('data/nba_advanced_1980_2026.csv')
flow = pd.read_csv('results/sample_flow.csv'); meta = pd.read_csv('results/players_primary.csv')
d = np.load('results/trajectories_primary.npz'); X, M = d['X'], d['M']
num('nRawRows', fmtn(len(raw))); num('nPlayerSeasons', fmtn(len(ps))); num('nAllPlayers', fmtn(ps.player_id.nunique()))
fp = flow[flow.spec == 'primary'].set_index('step').n
num('nDebutWindow', fmtn(fp['debut season in window'])); num('nGamesFortytwo', fmtn(fp['>= 42 career games']))
num('nPlayers', fmtn(fp['final players'])); num('nObs', fmtn(fp['valid PER observations']))
num('nActive', str(int((meta['last'] == 2026).sum()))); num('minActiveSeason', str(int(meta.loc[meta['last'] == 2026, 'n_valid'].min())))
num('sdPerLowMp', ps[ps.mp < 50].per.std(), '{:.1f}'); num('sdPerHighMp', ps[ps.mp >= 500].per.std(), '{:.1f}')
num('minSpan', str(int(meta.span.min()))); num('maxSpan', str(int(meta.span.max())))
num('nGapPlayers', fmtn(fp['players with >=1 missing season inside span'])); num('pctGapPlayers', 100 * fp['players with >=1 missing season inside span'] / fp['final players'], '{:.1f}')
num('nGapSeasons', fmtn(meta.n_gaps.sum())); num('nSpanSeasons', fmtn(meta.span.sum())); num('pctGapSeasons', 100 * meta.n_gaps.sum() / meta.span.sum(), '{:.1f}')
aej = json.load(open('results/ae_primary_d64_s0.json'))['metrics']
num('muPer', aej['mu']); num('sdPer', aej['sd']); num('skewPer', stats.skew(X[M]), '{:.2f}')
num('nDev', fmtn(aej['n_train'])); num('nHeld', fmtn(aej['n_val']))
dm = pd.read_csv('results/draft_match.csv'); num('nDrafted', fmtn(dm.set_index('player_id').loc[meta.player_id].pick.notna().sum()))
_f = ps.groupby('player_id').season_end.min(); _t = dm.set_index('player_id').first_nba_season.reindex(_f.index)
num('nVeterans', fmtn(((_f >= 1980) & (_f <= 2011) & (_t < 1980)).sum()))
_pc = meta.pos_mode.value_counts(); num('posMin', str(int(_pc.min()))); num('posMax', str(int(_pc.max()))); num('medianSeasons', str(int(meta.n_valid.median()))); num('meanSeasons', meta.n_valid.mean(), '{:.1f}')
num('nGrid', str(len(pd.read_csv('results/grid_primary_d64_s0.csv'))))
R = resample(X, M); from sklearn.decomposition import PCA
num('pcaVar', 100 * PCA(6).fit(((R - R.mean(0)) / R.std(0))).explained_variance_ratio_.sum(), '{:.0f}')

# ---------------- Table: sample flow ----------------
specs = {'primary': 'Primary', 'original2025': 'No minutes threshold, PER$>$45 removed', 'mp250': '250-minute threshold', 'min3seasons': 'At least 3 valid seasons', 'nogames': 'No career-games restriction', 'debut1980_2005': 'Debut 1979--80 to 2005--06'}
rows = []
for s, lbl in specs.items():
    f = flow[flow.spec == s].set_index('step').n
    rows.append(f"{lbl} & {fmtn(f.iloc[0])} & {fmtn(f.iloc[1])} & {fmtn(f.iloc[2])} & {fmtn(f['valid PER observations'])} \\\\")
open('paper/tables/flow.tex', 'w').write("\\begin{tabular}{lrrrr}\\toprule\nSpecification & Debut in window & Career games & Valid seasons & PER observations \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ---------------- Table: AE config ----------------
cfg = aej['config']
open('paper/tables/aeconfig.tex', 'w').write(r"""\begin{tabular}{ll}\toprule
Component & Setting \\ \midrule
Input per season & $[x_{i,t}m_{i,t},\,m_{i,t}]$ (standardised PER, observed indicator) \\
Model width / heads / feed-forward & 64 / 4 / 128 \\
Encoder / decoder layers & 2 / 2 (post-norm, ReLU, dropout 0.1) \\
Positional encoding & learned, season index (encoder) and output position (decoder) \\
Latent dimensionality $d$ & 8, 16, 32, 64, 128 (linear projection of the final classification-token (CLS) state) \\
Initialisation & Xavier-uniform weights, zero biases, CLS $\sim\mathcal N(0,0.02^2)$ \\
Loss & masked MSE, Eq.~\eqref{M-eq:loss} of the main text \\
Optimiser & AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$ \\
Batch size / gradient clipping & 64 / global norm 1.0 \\
Schedule & halve learning rate after 15 epochs without validation improvement \\
Epochs / early stopping & at most 400; patience 40 on held-out masked MSE; best epoch restored \\
Data split & 80/20 by player, stratified by career-length tertile, split seed 2026 \\
Seeds & 0--4 ($d=64$), 0--2 (other $d$); deterministic algorithms enforced \\
Parameters & """ + fmtn(aej['n_params']) + r""" ($d=64$) \\
Hardware / time & 2 CPU cores; """ + f"{aej['train_seconds']/60:.0f}" + r""" min per model ($d=64$) \\
\bottomrule\end{tabular}""")

# ---------------- Table: reconstruction ----------------
rows = []
for f in [f_ for f_ in glob.glob('results/ae_primary_d*_s*.json') if '_3way' not in f_]: rows.append(json.load(open(f))['metrics'])
ae = pd.DataFrame(rows); g = ae.groupby('latent').agg(n=('seed', 'size'), tr=('rmse_train', 'mean'), va=('rmse_val', 'mean'), vsd=('rmse_val', 'std'), ep=('best_epoch', 'mean')).reset_index()
b = ae.iloc[0]
lines = [f"Transformer autoencoder, $d={int(r.latent)}$ & {int(r.n)} & {r.tr:.2f} & {r.va:.2f} ({0 if np.isnan(r.vsd) else r.vsd:.2f}) & {r.ep:.0f} \\\\" for r in g.itertuples()]
lines += [r"\midrule Player-specific linear trend (2 parameters) & -- & -- & " + f"{b.baseline_player_linear_val:.2f}" + r" & -- \\",
          r"Player mean & -- & -- & " + f"{b.baseline_player_mean_val:.2f}" + r" & -- \\",
          r"Global mean & -- & -- & " + f"{b.baseline_global_mean_val:.2f}" + r" & -- \\"]
open('paper/tables/recon.tex', 'w').write("\\begin{tabular}{lcccc}\\toprule\nModel & Seeds & Train RMSE & Held-out RMSE (SD over seeds) & Best epoch \\\\ \\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}")
r64 = g[g.latent == 64].iloc[0]
num('rmseHeld', r64.va); num('rmseHeldSd', r64.vsd); num('rmseTrain', r64.tr); num('rmseLinear', b.baseline_player_linear_val); num('rmsePlayerMean', b.baseline_player_mean_val); num('rmseGlobal', b.baseline_global_mean_val)
num('rmseHeldEight', g[g.latent == 8].iloc[0].va); num('rmseHeldSixteen', g[g.latent == 16].iloc[0].va); num('rmseHeldThirtytwo', g[g.latent == 32].iloc[0].va); num('rmseHeldOneTwoEight', g[g.latent == 128].iloc[0].va)
num('bestEpoch', r64.ep, '{:.0f}'); num('nParams', fmtn(aej['n_params']))
num('rmseOld', '2.23')

# ---------------- Table: single-run selection (grid) ----------------
sel = json.load(open('results/selection_primary_d64_s0.json'))
rows = []
names = {'dbcv': 'DBCV (prespecified)', 'silhouette': 'Silhouette', 'composite_2025': 'Composite rule'}
for rule, r in sel.items():
    c = r['config']; cfgs = f"{int(c['n_components'])}/{int(c['n_neighbors'])}/{c['min_dist']:.1f}/{int(c['min_cluster_size'])}/{int(c['min_samples'])}"
    rows.append(f"{names[rule]} & {cfgs} & {r['dev']['n_clusters']} & {100*r['dev']['noise']:.0f} & {r['dev']['silhouette']:.2f} & {r['dev']['dbcv']:.2f} & {r['heldout']['n_clusters']} & {100*r['heldout']['noise']:.0f} & {r['heldout']['silhouette']:.2f} & {r['all']['n_clusters']} & {100*r['all']['noise']:.0f} \\\\")
open('paper/tables/selection.tex', 'w').write("\\begin{tabular}{llccccccccc}\\toprule\n & & \\multicolumn{4}{c}{Development (80\\%)} & \\multicolumn{3}{c}{Held-out (20\\%)} & \\multicolumn{2}{c}{Refit on all} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\nSelection rule & Configuration$^a$ & $k$ & Noise \\% & Sil. & DBCV & $k$ & Noise \\% & Sil. & $k$ & Noise \\% \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
short = {'dbcv': 'Dbcv', 'silhouette': 'Sil', 'composite_2025': 'Comp'}
for rule, r in sel.items():
    num(f'sel{short[rule]}Kdev', str(r['dev']['n_clusters'])); num(f'sel{short[rule]}Kho', str(r['heldout']['n_clusters'])); num(f'sel{short[rule]}Kall', str(r['all']['n_clusters']))
    num(f'sel{short[rule]}Noiseho', 100 * r['heldout']['noise'], '{:.0f}')
grid = pd.read_csv('results/grid_primary_d64_s0.csv'); num('gridKmin', str(int(grid.n_clusters.min()))); num('gridKmax', str(int(grid.n_clusters.max())))

# ---------------- Table: single-run stability ----------------
if os.path.exists('results/stability_single_run.csv'):
    st = pd.read_csv('results/stability_single_run.csv')
    cfgn = {c: f'C{i+1}' for i, c in enumerate(st.config.unique())}
    g = st.groupby(['config', 'method']).agg(ari=('ari_mean', 'mean'), amin=('ari_min', 'min'), amax=('ari_max', 'max'), kmin=('k_min', 'min'), kmed=('k_median', 'median'), kmax=('k_max', 'max'), noise=('noise_mean', 'mean'), big=('largest_cluster_share', 'mean')).reset_index()
    ac = pd.read_csv('results/stability_across_ae_seeds.csv')
    rows = []
    for r in g.itertuples():
        a2 = ac[(ac.config == r.config) & (ac.method == r.method)].iloc[0]
        rows.append(f"{cfgn[r.config]} & {'excess of mass' if r.method == 'eom' else r.method} & {r.ari:.2f} ({r.amin:.2f}--{r.amax:.2f}) & {a2.ari_mean_across_ae_seeds:.2f} & {int(r.kmin)}--{int(r.kmax)} ({r.kmed:.0f}) & {100*r.noise:.0f} & {100*r.big:.0f} \\\\")
    rows = sorted(rows, key=lambda r_: (r_.split(' & ')[0], r_.split(' & ')[1]))
    open('paper/tables/stability.tex', 'w').write("\\begin{tabular}{llccccc}\\toprule\nConfig.$^a$ & Extraction & ARI, UMAP seeds (range) & ARI, autoencoder seeds & $k$ range (median) & Noise \\% & Largest cluster \\% \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    num('ariSingleMin', g.ari.min()); num('ariSingleMax', g.ari.max()); num('ariAeSeedsMin', ac.ari_mean_across_ae_seeds.min()); num('ariAeSeedsMax', ac.ari_mean_across_ae_seeds.max())
    num('kSingleMin', str(int(g.kmin.min()))); num('kSingleMax', str(int(g.kmax.max()))); num('bigShareEomMax', 100 * g[g.method == 'eom'].big.max(), '{:.0f}'); num('bigShareEomMin', 100 * g[g.method == 'eom'].big.min(), '{:.0f}')
    num('noiseLeafMean', 100 * g[g.method == 'leaf'].noise.mean(), '{:.0f}'); num('noiseEomMean', 100 * g[g.method == 'eom'].noise.mean(), '{:.0f}')
    num('configList', '; '.join(f"{v}: {k.replace('(', '').replace(')', '').replace(' ', '')}" for k, v in cfgn.items()))

json.dump(N, open('results/numbers_partial.json', 'w'), indent=1)
assert all(k.isalpha() for k in N), [k for k in N if not k.isalpha()]
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
print(len(N), 'numbers written')

# =====================================================================================================
# Part 2: consensus clustering, baselines, typology, external validation, sensitivity
# =====================================================================================================
from evaluate import load_external, evaluate_partition, cramers_v, kw_epsilon2
KM = int(pd.read_csv('results/consensus_summary_latent.csv').query('selected').k.iloc[0])   # rule-selected k of the main representation
def labk(z, k): return z['labels_by_k'][list(z['k_list']).index(k)]
def stabk(z, k): return z['stability_by_k'][list(z['k_list']).index(k)]
def cons_summary(tag):
    df = pd.read_csv(f'results/consensus_{tag}.csv'); sel = df[df.selected].iloc[0]; k8 = df[df.k == KM].iloc[0]; k3 = df[df.k == 3].iloc[0]
    return df, sel, k8, k3
MAIN = 'summary_latent'
reps = [('ae64_latent', 'AE $d=64$, 5 seeds'), ('ae64_latent_seed0only', 'AE $d=64$, seed 0'),
        ('ae64_latent_nostd', 'AE $d=64$, unstandardised'), ('ae64_umap5', 'AE $d=64$ + UMAP'),
        ('ae64_umap5_hdbscan', 'AE + UMAP + HDBSCAN'),
        ('ae8_latent', 'AE $d=8$'), ('ae16_latent', 'AE $d=16$'), ('ae32_latent', 'AE $d=32$'), ('ae128_latent', 'AE $d=128$'),
        ('summary_latent', 'Summary statistics'), ('summary_latent_ward', 'Summary statistics, Ward'), ('raw_latent', 'Resampled trajectory'),
        ('pca_latent', 'PCA'), ('dtw_latent', 'DTW')]
m = load_external(); R = resample(X, M)
rows = []; EV = {}
for tag, lbl in reps:
    if not os.path.exists(f'results/consensus_{tag}.csv'): continue
    df, sel, k8, k3 = cons_summary(tag); z = np.load(f'results/consensus_{tag}.npz')
    e_sel = evaluate_partition(z['labels'], m, R); e8 = evaluate_partition(labk(z, KM), m, R); EV[tag] = (sel, k8, e_sel, e8, z)
    kk = f"{int(sel.k)}$^a$" if int(sel.k) == df.k.max() else str(int(sel.k))
    if tag == 'dtw_latent': kk += '$^b$'
    rows.append(f"{lbl} & {int(sel.members)} & {kk} & {sel.silhouette_consensus:.2f} & {sel.pac:.2f} & {100*sel.frac_stable:.0f} & {e_sel['shape_r2']:.2f} & {k8.silhouette_consensus:.2f} & {k8.pac:.2f} & {100*k8.frac_stable:.0f} & {e8['shape_r2']:.2f} & {e8['cramersV_all_star_ever']:.2f} & {e8['eps2_sum_ws']:.2f} \\\\")
open('paper/tables/baselines.tex', 'w').write("\\begin{tabular}{lcccccccccccc}\\toprule\n & & \\multicolumn{5}{c}{$k$ selected by the rule} & \\multicolumn{6}{c}{Matched, $k=" + str(KM) + "$} \\\\ \\cmidrule(lr){3-7}\\cmidrule(lr){8-13}\nRepresentation & Members & $k$ & Sil.$_C$ & PAC & Stable & $R^2_{\\mathrm{shape}}$ & Sil.$_C$ & PAC & Stable & $R^2_{\\mathrm{shape}}$ & $V_{\\mathrm{AS}}$ & $\\varepsilon^2_{\\mathrm{WS}}$ \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
def tagnum(tag, short):
    if tag not in EV: return
    sel, k8, e_sel, e8, z = EV[tag]
    num(f'{short}K', str(int(sel.k))); num(f'{short}Sil', sel.silhouette_consensus); num(f'{short}Pac', sel.pac); num(f'{short}Stable', 100 * sel.frac_stable, '{:.0f}')
    num(f'{short}SilEight', k8.silhouette_consensus); num(f'{short}PacEight', k8.pac); num(f'{short}StableEight', 100 * k8.frac_stable, '{:.0f}'); num(f'{short}RsqEight', e8['shape_r2']); num(f'{short}Rsq', e_sel['shape_r2'])
for tag, short in [('ae64_latent', 'aeAll'), ('ae64_latent_seed0only', 'aeSeedZero'), ('ae64_latent_nostd', 'aeNostd'), ('ae64_umap5', 'aeUmap'), ('ae64_umap5_hdbscan', 'aeHdb'), ('summary_latent', 'summ'), ('summary_latent_ward', 'summWard'), ('raw_latent', 'raw'), ('pca_latent', 'pca'), ('dtw_latent', 'dtw'), ('ae8_latent', 'aeEight'), ('ae16_latent', 'aeSixteen'), ('ae32_latent', 'aeThirtytwo'), ('ae128_latent', 'aeOneTwoEight')]:
    tagnum(tag, short)
# consensus-curve facts
dfm = pd.read_csv(f'results/consensus_{MAIN}.csv'); num('summSilTwo', dfm[dfm.k == 2].silhouette_consensus.iloc[0]); num('summSilFour', dfm[dfm.k == 4].silhouette_consensus.iloc[0]); num('summSilTwelve', dfm[dfm.k == 12].silhouette_consensus.iloc[0])
dfa = pd.read_csv('results/consensus_ae64_latent.csv'); num('aeAllSilThree', dfa[dfa.k == 3].silhouette_consensus.iloc[0]); num('aeAllSilTwo', dfa[dfa.k == 2].silhouette_consensus.iloc[0]); num('aeAllSilTwelve', dfa[dfa.k == 12].silhouette_consensus.iloc[0]); num('aeAllPacThree', dfa[dfa.k == 3].pac.iloc[0])
num('nMembersAe', str(int(dfa.members.iloc[0]))); num('nMembersSumm', str(int(dfm.members.iloc[0])))
num('nMembersDtw', str(int(pd.read_csv('results/consensus_dtw_latent.csv').members.iloc[0])) if os.path.exists('results/consensus_dtw_latent.csv') else '8')

# ---------------- typology: MAIN representation, k=4 (rule) and k=8 (fine) ----------------
zm = np.load(f'results/consensus_{MAIN}.npz'); lab8 = labk(zm, KM); stab8 = stabk(zm, KM)
num('mainK', str(KM)); _ok = dfm[dfm.k >= 2]
KC = 2; num('coarseK', str(KC))              # two-group split, used only for comparisons with the preliminary and autoencoder partitions
_ix = list(zm['k_list']).index(KC); lab4 = zm['labels_by_k'][_ix]; stab4 = zm['stability_by_k'][_ix]
num('coarseSil', float(_ok[_ok.k == KC].silhouette_consensus.iloc[0])); num('coarsePac', float(_ok[_ok.k == KC].pac.iloc[0]))
num('summPacMin', float(dfm[dfm.k >= 3].pac.min()))
feats = summary_features(X, M); m2 = m.copy(); m2['slope'] = feats['slope'].values; m2['peak_pos'] = feats['peak_pos'].values
def order_clusters(lab):
    g = pd.DataFrame(dict(lab=lab, mp=m2.mean_per, n=m2.n_valid)).groupby('lab').agg(mp=('mp', 'median'), n=('n', 'median'))
    return list(g.sort_values(['mp', 'n'], ascending=[False, False]).index)
def label_rule(mp, n, slope):
    level = 'high' if mp > 18 else 'moderate' if mp >= 12 else 'low'
    dur = 'long' if n >= 10 else 'medium' if n >= 5 else 'short'
    shape = ', rising' if slope > 0.25 else ', declining' if slope < -0.25 else ''
    return f'{level}-PER, {dur} career{shape}'
def profile_table(lab, stab, fname, prefix, letter='G'):
    order = order_clusters(lab); rows = []; labels = []; meds = []
    for j, c in enumerate(order):
        ii = lab == c; g = m2[ii]
        lbl = label_rule(g.mean_per.median(), g.n_valid.median(), g.slope.median()); labels.append(lbl)
        # medoid-nearest examples: closest to cluster median in (mean_per, n_valid, slope) z-space
        Fz = (feats[['mean', 'n_seasons', 'slope', 'peak']] - feats[['mean', 'n_seasons', 'slope', 'peak']].mean()) / feats[['mean', 'n_seasons', 'slope', 'peak']].std()
        dist = ((Fz[ii] - Fz[ii].median()) ** 2).sum(1); ex = m2.loc[dist.sort_values().index[:3], 'name'].tolist()
        rows.append(f"{letter}{j+1} & {ii.sum()} & {np.median(stab[ii]):.2f} & {g.mean_per.median():.1f} & {g.peak_per.median():.1f} & {int(g.n_valid.median())} & {('$-$' + format(-g.slope.median(), '.2f')) if g.slope.median() < 0 else format(g.slope.median(), '+.2f')} & {100*(g.all_star_n>0).mean():.0f} & {g.sum_ws.median():.1f} & {lbl} \\\\"); meds.append(f"{letter}{j+1}: {', '.join(ex)}")
    open(f'paper/tables/{fname}.tex', 'w').write("\\begin{tabular}{lcccccccc>{\\raggedright\\arraybackslash}p{4.4cm}}\\toprule\nGroup & $n$ & Stability & Mean PER & Peak PER & Seasons & Slope & All-Star \\% & Career WS & Label (rule) \\\\ \\midrule\n" + "\n".join(rows) + "\n\\midrule\n\\multicolumn{10}{p{0.97\\linewidth}}{\\raggedright Players nearest to the group medoid. " + '; '.join(meds) + ".} \\\\\n\\bottomrule\\end{tabular}")
    for j, c in enumerate(order):
        ii = lab == c; g = m2[ii]
        num(f'{prefix}N{"ABCDEFGHIJKL"[j]}', str(int(ii.sum()))); num(f'{prefix}Per{"ABCDEFGHIJKL"[j]}', g.mean_per.median(), '{:.1f}'); num(f'{prefix}Seas{"ABCDEFGHIJKL"[j]}', str(int(g.n_valid.median())))
        num(f'{prefix}AS{"ABCDEFGHIJKL"[j]}', 100 * (g.all_star_n > 0).mean(), '{:.0f}'); num(f'{prefix}Stab{"ABCDEFGHIJKL"[j]}', np.median(stab[ii])); num(f'{prefix}Slope{"ABCDEFGHIJKL"[j]}', g.slope.median(), '{:+.2f}'); num(f'{prefix}SlopeX{"ABCDEFGHIJKL"[j]}', g.slope.median(), '{:+.3f}')
    return order, labels
order4, labels4 = profile_table(lab4, stab4, 'profilescoarse', 'coarse', letter='C'); order8, labels8 = profile_table(lab8, stab8, 'profiles8', 'eight')
json.dump(order8, open('results/cluster_order.json', 'w')); json.dump(labels8, open('results/cluster_labels.json', 'w')); json.dump(dict(order4=[int(x) for x in order4], labels4=labels4), open('results/cluster_labels4.json', 'w'))
# nesting of k=8 within k=4
ct = pd.crosstab(pd.Series(lab8).map({c: j + 1 for j, c in enumerate(order8)}), pd.Series(lab4).map({c: j + 1 for j, c in enumerate(order4)}))
open('paper/tables/nesting.tex', 'w').write(ct.to_latex().replace('col_0', 'Coarse group').replace('row_0', 'Group'))
num('ariFourEight', ari(lab4, lab8)); num('nestingPurity', 100 * ct.max(1).sum() / ct.values.sum(), '{:.0f}')
# named players
def where(name, lab, order):
    i = m2.index[m2.name == name]
    if len(i) == 0: return '?'
    return f"G{order.index(lab[i[0]]) + 1}"
for nm, key in [('Michael Jordan', 'Jordan'), ('Steve Nash', 'Nash'), ('Ralph Sampson', 'Sampson'), ('Darko Miličić', 'Milicic'), ('Adam Morrison', 'Morrison'), ('Kobe Bryant', 'Bryant'), ('Derrick Rose', 'Rose'), ('Isiah Thomas', 'Isiah'), ('LeBron James', 'LeBron'), ('Stephen Curry', 'Curry'), ('Larry Bird', 'Bird'), ('Manu Ginóbili', 'Ginobili'), ('Ben Wallace', 'Wallace')]:
    num(f'grp{key}', where(nm, lab8, order8)); num(f'stab{key}', float(stab8[m2.index[m2.name == nm][0]]) if (m2.name == nm).any() else 0.0)

# ---------------- external validation table (k=8, MAIN) ----------------
labR = pd.Series(lab8).map({c: j for j, c in enumerate(order8)}).values
e8 = evaluate_partition(labR, m, R)
for var, key in [('pos_mode', 'Pos'), ('draft_cat', 'Draft'), ('all_star_ever', 'AllStar'), ('all_nba_ever', 'AllNba')]:
    num(f'V{key}', e8[f'cramersV_{var}']); num(f'p{key}', e8[f'p_{var}'], '{:.3f}'); num(f'ami{key}', e8[f'ami_{var}'])
for var, key in [('sum_ws', 'Ws'), ('mean_bpm', 'Bpm'), ('career_games', 'Games'), ('mean_per', 'Per'), ('n_valid', 'Len')]:
    num(f'eps{key}', e8[f'eps2_{var}'])
ct = pd.crosstab(labR + 1, m.draft_cat, normalize='index')[['lottery', 'first_round', 'second_round', 'undrafted']] * 100
ct2 = pd.crosstab(labR + 1, m.pos_mode, normalize='index')[['PG', 'SG', 'SF', 'PF', 'C']] * 100
g = m.assign(T=labR + 1).groupby('T').agg(AS=('all_star_ever', 'mean'), AN=('all_nba_ever', 'mean'), ws=('sum_ws', 'median'), bpm=('mean_bpm', 'median'), games=('career_games', 'median'))
rows = [f"G{t} & {ct.loc[t,'lottery']:.0f} & {ct.loc[t,'first_round']:.0f} & {ct.loc[t,'second_round']:.0f} & {ct.loc[t,'undrafted']:.0f} & {ct2.loc[t,'PG']:.0f}/{ct2.loc[t,'SG']:.0f}/{ct2.loc[t,'SF']:.0f}/{ct2.loc[t,'PF']:.0f}/{ct2.loc[t,'C']:.0f} & {100*g.loc[t,'AS']:.0f} & {100*g.loc[t,'AN']:.0f} & {g.loc[t,'ws']:.1f} & {g.loc[t,'bpm']:.1f} & {g.loc[t,'games']:.0f} \\\\" for t in range(1, KM + 1)]
foot = f"\\midrule Cram\\'er's $V$ / $\\varepsilon^2$ & \\multicolumn{{4}}{{c}}{{$V={e8['cramersV_draft_cat']:.2f}$}} & $V={e8['cramersV_pos_mode']:.2f}$ & $V={e8['cramersV_all_star_ever']:.2f}$ & $V={e8['cramersV_all_nba_ever']:.2f}$ & $\\varepsilon^2={e8['eps2_sum_ws']:.2f}$ & $\\varepsilon^2={e8['eps2_mean_bpm']:.2f}$ & $\\varepsilon^2={e8['eps2_career_games']:.2f}$ \\\\"
open('paper/tables/external.tex', 'w').write("\\begin{tabular}{lccccccccccc}\\toprule\n & \\multicolumn{4}{c}{Draft category (\\% of group)} & Position & \\multicolumn{2}{c}{Honours (\\%)} & \\multicolumn{3}{c}{Career value (median)} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}\nGroup & Lottery & 1st rd & 2nd rd & Undrafted & PG/SG/SF/PF/C & All-Star & All-NBA & WS & BPM & Games \\\\ \\midrule\n" + "\n".join(rows) + "\n" + foot + "\n\\bottomrule\\end{tabular}")

# ---------------- agreement between representations and sensitivity ----------------
pid_primary = d['player_id']
def load_labels(tag, spec, useK8):
    zb = np.load(f'results/consensus_{tag}.npz'); pb = np.load(f'results/trajectories_{spec}.npz')['player_id']
    return (labk(zb, KM) if useK8 else zb['labels']), pb, int(zb['k'])
def ari_between(tagA, specA, tagB, specB, useK8=False):
    la, pa, ka = load_labels(tagA, specA, useK8); lb, pb, kb = load_labels(tagB, specB, useK8)
    common = np.intersect1d(pa, pb); ia = np.searchsorted(pa, common); ib = np.searchsorted(pb, common)
    return float(ari(la[ia], lb[ib])), len(common), ka, kb
def spec_of(tag):
    for sp in ['original2025', 'mp250', 'min3seasons', 'nogames', 'debut1980_2005', 'consecutive', 'primary_bpm']:
        if tag.endswith('_' + sp): return sp
    return 'primary'
# (a) sensitivity of the autoencoder pipeline, reference = ae64_latent (5 seeds)
sensA = []
for tag, lbl in [('ae64_latent_seed0only', 'Single AE seed'), ('ae64_latent_nostd', 'Unstandardised latent'), ('ae64_umap5', 'UMAP before $k$-means'), ('ae64_umap5_hdbscan', 'UMAP + HDBSCAN'),
                 ('ae8_latent', 'AE $d$=8'), ('ae16_latent', 'AE $d$=16'), ('ae32_latent', 'AE $d$=32'), ('ae128_latent', 'AE $d$=128'),
                 ('ae64_latent_original2025', 'No minutes threshold'), ('ae64_latent_mp250', '250-minute threshold'), ('ae64_latent_min3seasons', '$\\ge$3 seasons'), ('ae64_latent_nogames', 'No games rule'), ('ae64_latent_debut1980_2005', 'Debut $\\le$2005--06'), ('ae64_latent_consecutive', 'Gaps deleted'), ('ae64_latent_primary_bpm', 'BPM instead of PER')]:
    if not os.path.exists(f'results/consensus_{tag}.npz'): continue
    ref = 'ae64_latent_seed0only' if spec_of(tag) != 'primary' else 'ae64_latent'
    a, n, ka, kb = ari_between(ref, 'primary', tag, spec_of(tag)); a8, _, _, _ = ari_between(ref, 'primary', tag, spec_of(tag), True)
    sensA.append(dict(label=lbl, n=n, k=kb, ari_sel=a, ari_k8=a8, tag=tag))
# (b) sensitivity of the summary-statistic typology, reference = summary_latent
sensS = []
for tag, lbl in [('summary_latent_original2025', 'No minutes threshold'), ('summary_latent_mp250', '250-minute threshold'), ('summary_latent_min3seasons', '$\\ge$3 seasons'), ('summary_latent_nogames', 'No games rule'), ('summary_latent_debut1980_2005', 'Debut $\\le$2005--06'), ('summary_latent_consecutive', 'Gaps deleted'), ('summary_latent_primary_bpm', 'BPM instead of PER'), ('summary_latent_ward', 'Ward instead of $k$-means')]:
    if not os.path.exists(f'results/consensus_{tag}.npz'): continue
    a, n, ka, kb = ari_between('summary_latent', 'primary', tag, spec_of(tag)); a8, _, _, _ = ari_between('summary_latent', 'primary', tag, spec_of(tag), True)
    sensS.append(dict(label=lbl, n=n, k=kb, ari_sel=a, ari_k8=a8, tag=tag))
sensA = pd.DataFrame(sensA); sensS = pd.DataFrame(sensS)
rows = ["\\multicolumn{5}{l}{\\textit{(a) Autoencoder pipeline; reference: five-seed consensus (model variants) or seed-0 consensus (data variants, single seed)}} \\\\"] + [f"{r.label}{'$^a$' if spec_of(r.tag) != 'primary' else ''} & {int(r.n):,} & {r.k} & {r.ari_sel:.2f} & {r.ari_k8:.2f} \\\\" for r in sensA.itertuples()]
rows += ["\\midrule \\multicolumn{5}{l}{\\textit{(b) Summary-statistic segmentation; reference: primary consensus}} \\\\"] + [f"{r.label} & {int(r.n):,} & {r.k} & {r.ari_sel:.2f} & {r.ari_k8:.2f} \\\\" for r in sensS.itertuples()]
open('paper/tables/sensitivity.tex', 'w').write("\\begin{tabular}{lcccc}\\toprule\nVariant & Common players & Selected $k$ & ARI with reference (selected $k$) & ARI with reference ($k=" + str(KM) + "$) \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
sensS.to_csv('results/sensitivity_summary.csv', index=False); sensA.to_csv('results/sensitivity_ae.csv', index=False)
json.dump([[r.label, r.ari_k8] for r in sensS.itertuples()], open('results/sensitivity_ari.json', 'w'))
def keyof(t): return ''.join(ch for ch in t.replace('_latent', '').replace('_', ' ').title().replace(' ', '') if ch.isalpha())
for r in sensA.itertuples(): num(f'ariA{keyof(r.tag)}', r.ari_sel); num(f'ariAEight{keyof(r.tag)}', r.ari_k8); num(f'kA{keyof(r.tag)}', str(r.k))
for r in sensS.itertuples(): num(f'ariS{keyof(r.tag)}', r.ari_sel); num(f'ariSEight{keyof(r.tag)}', r.ari_k8); num(f'kS{keyof(r.tag)}', str(r.k))
if len(sensS): num('ariSMin', sensS.ari_sel.min()); num('ariSMax', sensS.ari_sel.max()); num('ariSEightMin', sensS.ari_k8.min()); num('ariSEightMax', sensS.ari_k8.max())
if len(sensA): num('ariAMin', sensA.ari_sel.min()); num('ariAMax', sensA.ari_sel.max())
# (c) cross-representation agreement at k=8
cross = ['summary_latent', 'raw_latent', 'pca_latent', 'dtw_latent', 'ae64_latent', 'ae64_latent_seed0only', 'ae64_umap5']
cross = [c for c in cross if os.path.exists(f'results/consensus_{c}.npz')]
A = np.eye(len(cross)); 
for i in range(len(cross)):
    for j in range(len(cross)):
        if i < j: A[i, j] = A[j, i] = ari_between(cross[i], 'primary', cross[j], 'primary', True)[0]
short = {'summary_latent': 'Summary', 'raw_latent': 'Resampled', 'pca_latent': 'PCA', 'dtw_latent': 'DTW', 'ae64_latent': 'AE (5 seeds)', 'ae64_latent_seed0only': 'AE (seed 0)', 'ae64_umap5': 'AE+UMAP'}
rows = [short[c] + ' & ' + ' & '.join(f'{A[i,j]:.2f}' if j <= i else '' for j in range(len(cross))) + ' \\\\' for i, c in enumerate(cross)]
open('paper/tables/crossrep.tex', 'w').write("\\begin{tabular}{l" + "c" * len(cross) + "}\\toprule\n & " + " & ".join(short[c] for c in cross) + " \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
for i, c in enumerate(cross):
    for j, c2 in enumerate(cross):
        if i < j: num(f'x{keyof(c)}{keyof(c2)}', A[i, j])
# AE coarse structure vs summary coarse structure
za = np.load('results/consensus_ae64_latent.npz'); num('ariAeSummCoarse', ari(za['labels'], lab4)); num('ariAeSummCoarseThree', ari(za['labels'], zm['labels_by_k'][1]))
json.dump(N, open('results/numbers.json', 'w'), indent=1)
assert all(k.isalpha() for k in N), [k for k in N if not k.isalpha()]
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
print(len(N), 'numbers written (part 2)')

# ---------------- Appendix tables: k-curve and internal indices ----------------
dS = pd.read_csv('results/consensus_summary_latent.csv'); dA = pd.read_csv('results/consensus_ae64_latent.csv')
rows = [f"{int(a.k)} & {a.pac:.2f} & {a.silhouette_consensus:.2f} & {100*a.frac_stable:.0f} & {b.pac:.2f} & {b.silhouette_consensus:.2f} & {100*b.frac_stable:.0f} \\\\" for a, b in zip(dS.itertuples(), dA.itertuples())]
open('paper/tables/kcurve.tex', 'w').write("\\begin{tabular}{ccccccc}\\toprule\n & \\multicolumn{3}{c}{Summary statistics} & \\multicolumn{3}{c}{Autoencoder $d=64$, 5 seeds} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n$k$ & PAC & Sil.$_C$ & Stable \\% & PAC & Sil.$_C$ & Stable \\% \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
spaces = {'summary_latent': StandardScaler().fit_transform(summary_features(X, M).values), 'raw_latent': StandardScaler().fit_transform(R),
          'pca_latent': PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(R)), 'ae64_latent': StandardScaler().fit_transform(np.load('results/ae_primary_d64_s0.npz')['Z'])}
rows = []
for tag, S in spaces.items():
    if tag not in EV: continue
    l8 = labk(EV[tag][4], KM)
    rows.append(f"{dict(reps)[tag]} & {silhouette_score(S, l8):.2f} & {calinski_harabasz_score(S, l8):.0f} & {davies_bouldin_score(S, l8):.2f} \\\\")
open('paper/tables/internal.tex', 'w').write("\\begin{tabular}{lccc}\\toprule\nRepresentation & Silhouette & Calinski--Harabasz & Davies--Bouldin \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')

# ---------------- structure tests, null references, remaining text macros ----------------
st = json.load(open('results/structure_tests.json'))
num('gapSummNormalMax', st['gap_max_summary_normal']); num('gapRawNormalMax', st['gap_max_raw_normal']); num('gapAeNormalMax', st['gap_max_ae64_s0_normal'])
num('gapSummUniformMax', st['gap_max_summary_uniform'])
num('gapKSummUniform', str(st['gap_k_summary_uniform'])); num('gapKRawUniform', str(st['gap_k_raw_uniform']))
dips = [v for k, v in st.items() if k.startswith('dip_p_') and 'pc' in k]; num('dipPmin', min(dips)); num('dipPmax', max(dips))
num('dipPseasons', st['dip_p_feat_n_seasons'], '{:.3f}')
def null_stats(prefix):
    rows = []
    for b in range(3):
        f = f'results/consensus_{prefix}{b}_latent.csv'
        if os.path.exists(f): rows.append(pd.read_csv(f))
    if not rows: return None
    dd = pd.concat(rows); g = dd.groupby('k').agg(sil=('silhouette_consensus', 'mean'), pac=('pac', 'mean'), stable=('frac_stable', 'mean'), silmax=('silhouette_consensus', 'max'))
    return g
NS = {p: null_stats(p) for p in ['summary_null', 'raw_null', 'summary_copula', 'raw_copula']}
real = {'summary': pd.read_csv('results/consensus_summary_latent.csv').set_index('k'), 'raw': pd.read_csv('results/consensus_raw_latent.csv').set_index('k'), 'ae': pd.read_csv('results/consensus_ae64_latent.csv').set_index('k')}
def rs(name, k, col): return real[name].loc[k, col]
rows = []
for lbl, name, gk, dk in [('Summary statistics', 'summary', 'summary', 'summary'), ('Resampled trajectory', 'raw', 'raw', 'raw'), ('Autoencoder $d=64$, 5 seeds$^a$', 'ae', 'ae64_s0', 'ae64_s0')]:
    gmax = st[f'gap_max_{gk}_normal']; gsel = st[f'gap_k_{gk}_normal']; dp = min(st[f'dip_p_{dk}_pc{j}'] for j in (1, 2, 3))
    n4 = n8 = c4 = c8 = '--'
    if name != 'ae':
        Ng = NS[f'{name}_null']; Cg = NS[f'{name}_copula']
        if Ng is not None: n4, n8 = f"{Ng.loc[KM,'sil']:.2f}", f"{Ng.loc[8,'sil']:.2f}"
        if Cg is not None: c4, c8 = f"{Cg.loc[KM,'sil']:.2f}", f"{Cg.loc[8,'sil']:.2f}"
    rows.append(f"{lbl} & {gmax:.2f} & {gsel if gsel else '--'} & {dp:.2f} & {rs(name,KM,'silhouette_consensus'):.2f} / {rs(name,8,'silhouette_consensus'):.2f} & {n4} / {n8} & {c4} / {c8} \\\\")
open('paper/tables/structure.tex', 'w').write("\\begin{tabular}{lcccccc}\\toprule\n & \\multicolumn{2}{c}{Gap statistic (normal ref.)} & Dip test & \\multicolumn{3}{c}{Consensus Silhouette, $k="+str(KM)+"$ / $k=8$} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-7}\nRepresentation & max gap & selected $k$ & min $p$ (PC1--3) & Real data & Gaussian null & Copula null \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
sn, cn = NS['summary_null'], NS['summary_copula']; rn, rc = NS['raw_null'], NS['raw_copula']
num('nullSummSilC', sn.loc[KM, 'sil']); num('nullSummSilEight', sn.loc[8, 'sil']); num('nullSummPacEight', sn.loc[8, 'pac'])
num('copSummSilC', cn.loc[KM, 'sil']); num('copSummSilEight', cn.loc[8, 'sil']); num('copSummPacEight', cn.loc[8, 'pac'])
num('nullRawSilEight', rn.loc[8, 'sil']); num('copRawSilEight', rc.loc[8, 'sil'])
num('summSilC', rs('summary', KM, 'silhouette_consensus')); num('rawSilC', rs('raw', KM, 'silhouette_consensus')); num('nullRawSilC', rn.loc[KM, 'sil']); num('copRawSilC', rc.loc[KM, 'sil']); num('summSilEightReal', rs('summary', 8, 'silhouette_consensus')); num('rawSilEightReal', rs('raw', 8, 'silhouette_consensus'))
def sel_ks(prefix):
    ks = []; pm = []
    for b_ in range(3):
        f = f'results/consensus_{prefix}{b_}_latent.csv'
        if os.path.exists(f):
            dd = pd.read_csv(f); r_ = dd[dd.selected].iloc[0]; ks.append(int(r_.k)); pm.append(float(r_.pac))
    return ks, pm
for prefix, key in [('summary_null', 'NullSumm'), ('summary_copula', 'CopSumm'), ('raw_null', 'NullRaw'), ('raw_copula', 'CopRaw')]:
    ks, pm = sel_ks(prefix); num(f'kSel{key}', ', '.join(str(k_) for k_ in ks)); num(f'pacSel{key}', min(pm))
for _k in ['dtwSil', 'dtwK', 'dtwSilEight']:
    if _k not in N: N[_k] = '--'
# cross-representation extremes
xs = {k: float(v) for k, v in N.items() if k.startswith('x') and k[1].isupper()}
num('xMin', min(xs.values())); num('xMax', max(xs.values()))
_diff = {k: v for k, v in xs.items() if not (k[1:].startswith('Ae') and 'Ae' in k[3:])}
num('xMaxDiff', max(_diff.values()))
if 'xSummaryRaw' not in N: num('xSummaryRaw', N.get('xSummaryRaw', '--'))
# lottery share in T1 and stability minimum
lab8o = pd.Series(lab8).map({c: j for j, c in enumerate(order8)}).values
mm8 = m.assign(T=lab8o)
num('eightLotteryA', 100 * (mm8[mm8['T'] == 0].draft_cat == 'lottery').mean(), '{:.0f}')
num('eightStabMin', min(float(N[f'eightStab{ch}']) for ch in 'ABCDEFGH'[:KM]))
num('eightASmaxOther', max(float(N[f'eightAS{ch}']) for ch in 'BCDEFGH'[:KM - 1]), '{:.0f}')
# sensitivity summary macros
if len(sensS):
    num('kSMin', str(int(sensS.k.min()))); num('kSMax', str(int(sensS.k.max())))
    bpm = sensS[sensS.tag == 'summary_latent_primary_bpm']
    if len(bpm): num('ariSEightBpm', bpm.ari_k8.iloc[0]); num('ariSBpm', bpm.ari_sel.iloc[0])
    nb = sensS[sensS.tag != 'summary_latent_primary_bpm']; num('ariSEightMinNoBpm', nb.ari_k8.min()); num('ariSEightMaxNoBpm', nb.ari_k8.max())
    rest = sensS[sensS.tag.isin(['summary_latent_original2025', 'summary_latent_mp250', 'summary_latent_min3seasons', 'summary_latent_debut1980_2005'])]
    num('ariSEightMinRestrict', rest.ari_k8.min()); num('ariSEightMaxRestrict', rest.ari_k8.max())
    num('corrPerBpm', np.corrcoef(meta.mean_per, meta.mean_bpm)[0, 1])
if len(sensA):
    for t, key in [('ae64_latent_seed0only', 'ariASeedZero'), ('ae8_latent', 'ariAdEight'), ('ae128_latent', 'ariAdOneTwoEight'), ('ae32_latent', 'ariAdThirtytwo'), ('ae16_latent', 'ariAdSixteen'), ('ae64_latent_primary_bpm', 'ariABpm')]:
        r = sensA[sensA.tag == t]
        if len(r): num(key, r.ari_sel.iloc[0])
assert all(k.isalpha() for k in N), [k for k in N if not k.isalpha()]
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
json.dump(N, open('results/numbers.json', 'w'), indent=1)
print(len(N), 'numbers written (final)')

# ---------------- continuous description and shape-only analysis ----------------
c = json.load(open('results/continuous.json'))
num('pcOneVar', 100 * c['var_ratio'][0], '{:.0f}'); num('pcTwoVar', 100 * c['var_ratio'][1], '{:.0f}'); num('pcThreeVar', 100 * c['var_ratio'][2], '{:.0f}')
num('pcOneMean', c['pc1_corr']['mean']); num('pcOneLen', c['pc1_corr']['n_seasons']); num('pcOnePeak', c['pc1_corr']['peak']); num('pcOneWs', c['pc1_corr_ws'])
num('pcTwoSlope', c['pc2_corr']['slope']); num('pcTwoPeakpos', c['pc2_corr']['peak_pos']); num('pcTwoMean', abs(c['pc2_corr']['mean']))
num('shapeVarOne', 100 * c['shape_var_ratio'][0], '{:.0f}'); num('shapeVarTwo', 100 * c['shape_var_ratio'][1], '{:.0f}')
num('shapeDipPmin', min(c['shape_dip_p_pc1'], c['shape_dip_p_pc2'])); num('dipSlopeP', c['dip_p_slope'])
num('peakFirstThird', 100 * c['frac_peak_first_third'], '{:.0f}'); num('peakLastThird', 100 * c['frac_peak_last_third'], '{:.0f}')
zs = np.load('results/consensus_shape_latent.npz'); cs = pd.read_csv('results/consensus_shape_latent.csv')
num('shapeK', str(int(zs['k']))); num('shapeSil', cs[cs.k == int(zs['k'])].silhouette_consensus.iloc[0]); num('shapeSilEight', cs[cs.k == 8].silhouette_consensus.iloc[0])
Fs = summary_features(X, M); Fs['lab'] = zs['labels']; Fs['as'] = meta.all_star_n > 0
g = Fs.groupby('lab').agg(n=('lab', 'size'), slope=('slope', 'median'), pp=('peak_pos', 'median'), ns=('n_seasons', 'median'), mean=('mean', 'median'), first=('first', 'median'), last=('last', 'median'), AS=('as', 'mean')).sort_values('pp')
rows = [f"S{j+1} & {int(r.n)} & {r.pp:.2f} & {r.slope:+.2f} & {r.first:.1f} & {r.last:.1f} & {r.mean:.1f} & {int(r.ns)} & {100*r.AS:.0f} \\\\" for j, r in enumerate(g.itertuples())]
open('paper/tables/shape.tex', 'w').write("\\begin{tabular}{lcccccccc}\\toprule\nShape group & $n$ & Peak position & Slope & First-season PER & Last-season PER & Mean PER & Seasons & All-Star \\% \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
for j, r in enumerate(g.itertuples()):
    num(f'shapeN{"ABC"[j]}', str(int(r.n))); num(f'shapePp{"ABC"[j]}', r.pp); num(f'shapeSeas{"ABC"[j]}', str(int(r.ns)))
num('shapeLenEps', kw_epsilon2([Fs.n_seasons.values[Fs.lab == c_] for c_ in np.unique(Fs.lab)])[2])
# three-way split check
tw = [json.load(open(f))['metrics'] for f in glob.glob('results/ae_primary_d64_s*_3way.json')]
if tw:
    num('nThreeWay', str(len(tw))); num('rmseTestThreeWay', np.mean([t['rmse_test'] for t in tw])); num('rmseTestThreeWaySd', np.std([t['rmse_test'] for t in tw])); num('rmseValThreeWay', np.mean([t['rmse_val'] for t in tw]))
assert all(k.isalpha() for k in N), [k for k in N if not k.isalpha()]
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
json.dump(N, open('results/numbers.json', 'w'), indent=1); print(len(N), 'numbers (continuous)')
# how well do level, tilt and length reproduce the 8 types? (cross-validated linear discriminant accuracy)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
zc = np.load('results/continuous_scores.npz'); Sc = np.column_stack([zc['S'][:, :2], np.log(meta.n_valid)])
acc = cross_val_score(LinearDiscriminantAnalysis(), Sc, lab8, cv=10).mean(); num('ldaAccThree', 100 * acc, '{:.0f}')
acc2 = cross_val_score(LinearDiscriminantAnalysis(), zc['S'][:, :2], lab8, cv=10).mean(); num('ldaAccTwo', 100 * acc2, '{:.0f}')
num('ldaChance', 100 * np.bincount(np.asarray(lab8).astype(int) - np.asarray(lab8).astype(int).min()).max() / len(lab8), '{:.0f}')
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n'); print('lda', acc, acc2)

# ---------------- finite Gaussian mixtures calibrated against copula references ----------------
mx = json.load(open('results/mixture_tests.json'))
labels_mx = [('pc12', 'Level, tilt'), ('coords', 'Level, tilt, length'), ('coords5', 'Level, tilt, length ($\\ge$5 seasons)'),
             ('summary', 'Summary statistics'), ('summary5', 'Summary statistics ($\\ge$5 seasons)'),
             ('bpm_pc12', 'BPM: level, tilt'), ('bpm_coords', 'BPM: level, tilt, length')]
rows = []
for key, lbl in labels_mx:
    if key not in mx: continue
    r = mx[key]; re_ = r['real']; kk = f"{re_['k_bic']}$^a$" if re_['k_bic'] == 10 else str(re_['k_bic'])
    ent = '--' if re_['rel_entropy'] is None or (isinstance(re_['rel_entropy'], float) and np.isnan(re_['rel_entropy'])) else f"{re_['rel_entropy']:.2f}"
    ne = np.nanmedian(r['null_entropy']); nk = int(np.median(r['null_k']))
    nsk = r['ns']['k_bic']
    rows.append(f"{lbl} & {kk} & {re_['dbic']:.0f} & {np.median(r['null_dbic']):.0f} & {r['p_dbic']:.2f} & {ent} & {ne:.2f} & {re_['appa_min']:.2f} & {nk} & {nsk} & {r['lrt_ns']['p']:.2f} \\\\")
    short_ = {'pc12': 'PcTwo', 'coords': 'Coords', 'coords5': 'CoordsFive', 'summary': 'Summ', 'summary5': 'SummFive', 'bpm_pc12': 'BpmPcTwo', 'bpm_coords': 'BpmCoords'}[key]
    num(f'mx{short_}K', str(re_['k_bic'])); num(f'mx{short_}Dbic', f"{re_['dbic']:,.0f}"); num(f'mx{short_}NullDbic', f"{np.median(r['null_dbic']):,.0f}")
    num(f'mx{short_}P', r['p_dbic']); num(f'mx{short_}Ent', ent); num(f'mx{short_}NullEnt', ne); num(f'mx{short_}Appa', re_['appa_min'])
    num(f'mx{short_}NullK', str(nk)); num(f'mx{short_}NsK', str(nsk)); num(f'mx{short_}PEnt', r['p_entropy'] if r['p_entropy'] is not None else 1.0)
    num(f'mx{short_}LrtP', r['lrt_ns']['p'])
open('paper/tables/mixture.tex', 'w').write("\\begin{tabular}{lcccccccccc}\\toprule\n & \\multicolumn{4}{c}{BIC} & \\multicolumn{3}{c}{Class separation} & \\multicolumn{2}{c}{Selected $k$} & LRT \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-11}\nRepresentation & $k$ & $\\Delta$BIC & Copula & $p$ & Entropy & Copula & APPA & Copula & NS & $p$ \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
md = json.load(open('results/mixture_diagnostics.json'))
num('mxDegAll', str(md['summary']['n_in_degenerate'])); num('mxDegAllPct', 100 * md['summary']['n_in_degenerate'] / len(meta), '{:.0f}')
num('mxTwoAll', str(md['summary']['n_pure_two'])); num('mxFracZeroFive', str(md['summaryFIVE'.replace('FIVE', '5')]['n_pure_frac0']))
num('mxFracZeroFivePct', 100 * md['summary5']['share_frac0_sample'], '{:.0f}')
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
json.dump(N, open('results/numbers.json', 'w'), indent=1); print('mixture numbers written')

# ---------------- external validation conditional on level and length ----------------
import statsmodels.api as sm
Xc = pd.DataFrame({'level': m2.mean_per.values, 'loglen': np.log(m2.n_valid.values)})
G = pd.get_dummies(pd.Series(labR, name='T'), prefix='T', drop_first=True).astype(float)
def incr_ols(y):
    b = sm.OLS(y, sm.add_constant(Xc)).fit(); f = sm.OLS(y, sm.add_constant(pd.concat([Xc, G], axis=1))).fit()
    Fv, pv, _ = f.compare_f_test(b); return f.rsquared - b.rsquared, pv, b.rsquared
def incr_logit(y):
    b = sm.Logit(y, sm.add_constant(Xc)).fit(disp=0); f = sm.Logit(y, sm.add_constant(pd.concat([Xc, G], axis=1))).fit(disp=0, maxiter=200, method='bfgs')
    lr = 2 * (f.llf - b.llf); from scipy.stats import chi2; pv = chi2.sf(lr, G.shape[1])
    return f.prsquared - b.prsquared, pv, b.prsquared
d1, p1, r1 = incr_ols(np.log1p(m.sum_ws.clip(lower=0).values)); num('condWsDr', d1, '{:.3f}'); num('condWsP', p1, '{:.3f}'); num('condWsBase', r1)
d2, p2, r2 = incr_logit(m.all_star_ever.astype(float).values); num('condAsDr', d2, '{:.3f}'); num('condAsP', p2, '{:.3f}'); num('condAsBase', r2)
d3, p3, r3 = incr_logit((m.draft_cat == 'lottery').astype(float).values); num('condLotDr', d3, '{:.3f}'); num('condLotP', p3, '{:.3f}'); num('condLotBase', r3)
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
json.dump(N, open('results/numbers.json', 'w'), indent=1); print('conditional validation written')

# ---------------- latent geometry, power simulation, peak age ----------------
if os.path.exists('results/latent_geometry.json'):
    lg = json.load(open('results/latent_geometry.json'))
    num('lgPearson', lg['mean']['pearson']); num('lgPearsonMin', lg['range']['pearson'][0]); num('lgPearsonMax', lg['range']['pearson'][1])
    num('lgSpearman', lg['mean']['spearman']); num('lgKnn', lg['mean']['knn10']); num('lgKnnMin', lg['range']['knn10'][0]); num('lgKnnMax', lg['range']['knn10'][1])
    num('lgLinR', lg['mean']['linear_r2']); num('lgKmAri', lg['kmeans8_ari_between_seeds']['mean']); num('lgKmInitAri', lg['kmeans8_ari_between_inits_seed0']['mean'])
    num('lgRefSummPearson', lg['vs_reference']['summary']['pearson']); num('lgRefRawPearson', lg['vs_reference']['resampled']['pearson'])
    num('lgRefSummKnn', lg['vs_reference']['summary']['knn10']); num('lgRefRawKnn', lg['vs_reference']['resampled']['knn10'])
    if 'effective_dimension' in lg:
        ed = lg['effective_dimension']; num('lgEffLatent', ed['latent'], '{:.0f}'); num('lgEffSumm', ed['summary'], '{:.1f}'); num('lgEffRaw', ed['resampled'], '{:.1f}')
_parts = [f for f in ('results/power_a.json', 'results/power_b.json') if os.path.exists(f)]
if len(_parts) == 2:
    _pa = [json.load(open(f)) for f in _parts]
    json.dump(dict(rows=_pa[0]['rows'] + _pa[1]['rows'], summary={**_pa[0]['summary'], **_pa[1]['summary']}, reps=_pa[0]['reps'], n_ref=_pa[0]['n_ref']),
              open('results/power_simulation.json', 'w'), indent=1)
if os.path.exists('results/power_simulation.json'):
    pw = json.load(open('results/power_simulation.json'))['summary']
    for dlt, key in [('0.0', 'Zero'), ('1.5', 'OneFive'), ('2.0', 'Two'), ('2.5', 'TwoFive'), ('3.0', 'Three'), ('4.0', 'Four')]:
        if dlt in pw:
            for mtr, mk in [('detect_bic', 'Bic'), ('detect_ns', 'Ns'), ('detect_dip', 'Dip'), ('detect_appa', 'Appa'), ('detect_any', 'Any')]:
                num(f'pw{mk}{key}', 100 * pw[dlt][mtr], '{:.0f}')
    dd = json.load(open('results/dip_direction.json')) if os.path.exists('results/dip_direction.json') else {}
    def _dbest(dlt):
        key = str(float(dlt)).rstrip('0').rstrip('.') if float(dlt) != int(float(dlt)) else str(int(float(dlt)))
        v = dd.get(key, dd.get(dlt)); return f"{100*v['dip_best_direction']:.0f}" if v else '--'
    for dlt, key in [('0.0', 'Zero'), ('3.0', 'Three'), ('4.0', 'Four')]:
        num(f'pwDipBest{key}', _dbest(dlt))
    rows = [f"{float(dlt):.1f} & {100*v['detect_bic']:.0f} & {100*v['detect_ns']:.0f} & {100*v['detect_dip']:.0f} & {_dbest(dlt)} \\\\" for dlt, v in sorted(pw.items(), key=lambda x: float(x[0]))]
    open('paper/tables/power.tex', 'w').write("\\begin{tabular}{ccccc}\\toprule\n & \\multicolumn{2}{c}{Gaussian mixtures} & \\multicolumn{2}{c}{Dip test} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nSeparation $\\Delta$ & BIC, calibrated & BIC, normal scores & First axis & Best of 36 axes \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
cj = json.load(open('results/continuous.json'))
if 'peak_age' in cj:
    pa = cj['peak_age']; num('peakAgeN', f"{pa['n']:,}"); num('peakAgeMed', pa['median'], '{:.0f}'); num('peakAgeQa', pa['q25'], '{:.0f}'); num('peakAgeQb', pa['q75'], '{:.0f}')
    num('peakAgeDipP', pa['dip_p']); num('peakAgeCorrFirst', pa['corr_first_age'])
open('paper/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')
json.dump(N, open('results/numbers.json', 'w'), indent=1); print('extra analyses written')
# typographic minus signs in every generated table
import glob as _g, re as _re
for _f in _g.glob('paper/tables/*.tex'):
    _t = open(_f).read(); _t2 = _re.sub(r'(?<=[\s&(/])-(?=\d)', '$-$', _t)
    if _t2 != _t: open(_f, 'w').write(_t2)
