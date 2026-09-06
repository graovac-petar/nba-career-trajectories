"""Step 6 - all figures for the manuscript (300 dpi PNG + PDF) from the result files."""
import numpy as np, pandas as pd, json, glob, os, sys, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
sys.path.insert(0, 'code'); warnings.filterwarnings('ignore')
from features import resample
OUT = 'paper/figures'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
PAL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#aec7e8', '#ffbb78']
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); plt.close(fig)

d = np.load('results/trajectories_primary.npz'); X, M, pid = d['X'], d['M'], d['player_id']
meta = pd.read_csv('results/players_primary.csv')
MAIN = 'summary_latent'
cons = np.load(f'results/consensus_{MAIN}.npz'); lab = cons['labels8']; stab = cons['stability8']; K = 8
order = json.load(open('results/cluster_order.json')) if os.path.exists('results/cluster_order.json') else list(range(K))
# relabel clusters in the reporting order (1..K)
remap = {old: new for new, old in enumerate(order)}; labR = np.array([remap[l] for l in lab])

# ---------- Figure 1: representative trajectories with gaps ----------
def fig_examples():
    names = ['Michael Jordan', 'Steve Nash', 'Ralph Sampson', 'Darko Miličić']
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
    for ax, nm in zip(axes, names):
        i = meta.index[meta.name == nm]
        if len(i) == 0: continue
        i = i[0]; t = np.arange(1, M[i].sum() + M[i].size - M[i].size + 1)
        seasons = np.arange(int(meta.debut[i]), int(meta.debut[i]) + int(meta.span[i]))
        y = np.where(M[i][:len(seasons)], X[i][:len(seasons)], np.nan)
        ax.plot(seasons, y, 'o-', ms=3, color=PAL[0]); ax.axhline(15, ls=':', color='grey', lw=0.8)
        gaps = seasons[~M[i][:len(seasons)]]
        for g in gaps: ax.axvspan(g - 0.5, g + 0.5, color='#f0e0e0')
        ax.set_title(nm, fontsize=9); ax.set_xlabel('Season (end year)'); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel('PER'); save(fig, 'fig1_examples')

# ---------- Figure 3: reconstruction vs latent dimension ----------
def fig_recon():
    rows = []
    for f in glob.glob('results/ae_primary_d*_s*.json'):
        m = json.load(open(f))['metrics']; rows.append(m)
    df = pd.DataFrame(rows); g = df.groupby('latent').agg(rmse=('rmse_val', 'mean'), sd=('rmse_val', 'std'), n=('seed', 'size')).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(g.latent, g.rmse, yerr=g.sd.fillna(0), fmt='o-', color=PAL[0], capsize=3, label='Transformer autoencoder (held-out)')
    b = df.iloc[0]
    for key, lbl, c in [('baseline_player_linear_val', 'Player linear trend', PAL[1]), ('baseline_player_mean_val', 'Player mean', PAL[3]), ('baseline_global_mean_val', 'Global mean', PAL[9])]:
        ax.axhline(b[key], ls='--', color=c, lw=1, label=lbl)
    ax.set_xscale('log', base=2); ax.set_xticks(g.latent); ax.set_xticklabels(g.latent)
    ax.set_xlabel('Latent dimensionality $d$'); ax.set_ylabel('Held-out RMSE (PER units)'); ax.legend(fontsize=7, frameon=False)
    save(fig, 'fig3_reconstruction'); return df

# ---------- Figure 4: single-run instability ----------
def fig_single_run():
    df = pd.read_csv('results/stability_single_run.csv')
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    cfgs = df.config.unique()
    for j, method in enumerate(['eom', 'leaf']):
        sub = df[df.method == method]
        for i, c in enumerate(cfgs):
            s = sub[sub.config == c]
            axes[0].errorbar(i + (j - 0.5) * 0.2, s.ari_mean.mean(), yerr=[[s.ari_mean.mean() - s.ari_min.min()], [s.ari_max.max() - s.ari_mean.mean()]], fmt='o' if method == 'eom' else 's', color=PAL[j], capsize=2, label=method if i == 0 else None)
            axes[1].errorbar(i + (j - 0.5) * 0.2, s.k_median.median(), yerr=[[s.k_median.median() - s.k_min.min()], [s.k_max.max() - s.k_median.median()]], fmt='o' if method == 'eom' else 's', color=PAL[j], capsize=2, label=method if i == 0 else None)
    for ax in axes: ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f'C{i+1}' for i in range(len(cfgs))])
    axes[0].axhline(0.8, ls=':', color='grey'); axes[0].set_ylabel('Pairwise ARI between seeds'); axes[0].set_ylim(0, 1); axes[0].legend(frameon=False, fontsize=8, title='HDBSCAN extraction')
    axes[1].set_ylabel('Number of clusters (min, median, max)'); axes[0].set_xlabel('UMAP/HDBSCAN configuration'); axes[1].set_xlabel('UMAP/HDBSCAN configuration')
    save(fig, 'fig4_single_run')

# ---------- Figure 5: consensus curves and matrix ----------
def fig_consensus():
    fig = plt.figure(figsize=(10, 3.2)); gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.15])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    reps = [('ae64_latent', 'Autoencoder ($d$=64, 5 seeds)'), ('ae64_latent_seed0only', 'Autoencoder (seed 0)'), ('ae64_umap5', 'Autoencoder + UMAP'), ('ae64_umap5_hdbscan', 'AE + UMAP + HDBSCAN'), ('summary_latent', 'Summary statistics'), ('raw_latent', 'Resampled trajectory'), ('pca_latent', 'PCA'), ('dtw_latent', 'DTW $k$-means'), ('summary_copula0_latent', 'Copula null (summary)'), ('summary_null0_latent', 'Gaussian null (summary)')]
    for i, (r, l) in enumerate(reps):
        f = f'results/consensus_{r}.csv'
        if not os.path.exists(f): continue
        c = pd.read_csv(f); ls_ = '--' if 'null' in r else '-'; ax1.plot(c.k, c.pac, 'o' + ls_, ms=3, color=PAL[i], label=l); ax2.plot(c.k, c.silhouette_consensus, 'o' + ls_, ms=3, color=PAL[i], label=l)
    ax1.set_xlabel('Number of clusters $k$'); ax1.set_ylabel('PAC (lower = more reproducible)'); ax2.set_xlabel('Number of clusters $k$'); ax2.set_ylabel('Consensus Silhouette')
    ax1.legend(fontsize=6.5, frameon=False)
    C = cons['C8'].astype(np.float32); idx = np.argsort(labR, kind='stable')
    im = ax3.imshow(C[np.ix_(idx, idx)], cmap='Blues', vmin=0, vmax=1, interpolation='nearest'); ax3.set_xticks([]); ax3.set_yticks([])
    b = np.cumsum(np.bincount(labR));
    for x in b[:-1]: ax3.axhline(x, color='k', lw=0.4); ax3.axvline(x, color='k', lw=0.4)
    ax3.set_title(f'Co-association matrix, consensus $k$={K}', fontsize=9); plt.colorbar(im, ax=ax3, fraction=0.046)
    save(fig, 'fig5_consensus')

# ---------- Figure 6: cluster profiles ----------
def fig_profiles(labels_tex):
    ncol = 4; nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow), sharey=True, sharex=True); axes = axes.ravel()
    for c in range(K):
        ax = axes[c]; ii = np.where(labR == c)[0]
        T = int(np.percentile(meta.span[ii], 90))
        for i in ii[:150]:
            y = np.where(M[i], X[i], np.nan); ax.plot(np.arange(1, X.shape[1] + 1), y, color=PAL[c % len(PAL)], alpha=0.08, lw=0.6)
        med = [np.nanmedian(np.where(M[ii, t], X[ii, t], np.nan)) if M[ii, t].sum() >= 10 else np.nan for t in range(X.shape[1])]
        q1 = [np.nanpercentile(np.where(M[ii, t], X[ii, t], np.nan), 25) if M[ii, t].sum() >= 10 else np.nan for t in range(X.shape[1])]
        q3 = [np.nanpercentile(np.where(M[ii, t], X[ii, t], np.nan), 75) if M[ii, t].sum() >= 10 else np.nan for t in range(X.shape[1])]
        tt = np.arange(1, X.shape[1] + 1)
        ax.fill_between(tt, q1, q3, color=PAL[c % len(PAL)], alpha=0.25); ax.plot(tt, med, color='k', lw=1.6)
        ax.axhline(15, ls=':', color='grey', lw=0.8)
        ax.set_title(f'T{c+1} ($n$={len(ii)}, stability {np.median(stab[ii]):.2f})\n{labels_tex[c]}', fontsize=7)
        ax.set_xlim(1, 20); ax.set_ylim(0, 32)
    for c in range(K, len(axes)): axes[c].axis('off')
    for ax in axes[-ncol:]: ax.set_xlabel('Career season')
    for r in range(nrow): axes[r * ncol].set_ylabel('PER')
    save(fig, 'fig6_profiles')

# ---------- Figure 7: external validation ----------
def fig_external():
    m = meta.copy(); dm = pd.read_csv('results/draft_match.csv'); m = m.merge(dm, on='player_id', how='left')
    m['draft_cat'] = np.select([m['pick'] <= 14, m['pick'] <= 30, m['pick'] > 30], ['Lottery', '1st round', '2nd round'], 'Undrafted')
    m['T'] = labR + 1
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    ct = pd.crosstab(m['T'], m['draft_cat'], normalize='index')[['Lottery', '1st round', '2nd round', 'Undrafted']]
    ct.plot(kind='bar', stacked=True, ax=axes[0], color=PAL[:4], width=0.8, legend=True); axes[0].legend(fontsize=6, frameon=False, ncol=2); axes[0].set_ylabel('Share of players'); axes[0].set_xlabel('Trajectory type'); axes[0].set_title('Draft category', fontsize=9)
    ct2 = pd.crosstab(m['T'], m['pos_mode'], normalize='index')[['PG', 'SG', 'SF', 'PF', 'C']]
    ct2.plot(kind='bar', stacked=True, ax=axes[1], color=PAL[:5], width=0.8); axes[1].legend(fontsize=6, frameon=False, ncol=3); axes[1].set_xlabel('Trajectory type'); axes[1].set_title('Modal position', fontsize=9)
    g = m.groupby('T').agg(AS=('all_star_n', lambda x: (x > 0).mean()), AN=('all_nba_n', lambda x: (x > 0).mean()))
    g.plot(kind='bar', ax=axes[2], color=[PAL[1], PAL[3]], width=0.8); axes[2].legend(['All-Star ever', 'All-NBA ever'], fontsize=6, frameon=False); axes[2].set_xlabel('Trajectory type'); axes[2].set_title('Honours', fontsize=9)
    m.boxplot(column='sum_ws', by='T', ax=axes[3], grid=False, showfliers=False); axes[3].set_title('Career Win Shares', fontsize=9); axes[3].set_xlabel('Trajectory type'); axes[3].set_ylabel('WS')
    fig.suptitle('');
    for ax in axes: ax.tick_params(axis='x', rotation=0)
    save(fig, 'fig7_external')

# ---------- Figure 8: sensitivity ARI heatmap ----------
def fig_sensitivity(pairs):
    names = [p[0].replace('$\\ge$', '≥').replace('$\\le$', '≤').replace('$', '') for p in pairs]; A = np.array([[p[1] for p in pairs]])
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 1, 1.6))
    im = ax.imshow(A, cmap='Greens', vmin=0, vmax=1, aspect='auto'); ax.set_yticks([]); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    for j, v in enumerate(A[0]): ax.text(j, 0, f'{v:.2f}', ha='center', va='center', fontsize=7, color='k' if v < 0.6 else 'w')
    ax.set_title('ARI with the primary consensus partition', fontsize=8); save(fig, 'fig8_sensitivity')

if __name__ == '__main__':
    fig_examples(); fig_recon(); print('fig1, fig3 ok')
    if os.path.exists('results/stability_single_run.csv'): fig_single_run(); print('fig4 ok')
    fig_consensus(); print('fig5 ok')
    labels_tex = json.load(open('results/cluster_labels.json')) if os.path.exists('results/cluster_labels.json') else [''] * K
    fig_profiles(labels_tex); fig_external(); print('fig6, fig7 ok')
    if os.path.exists('results/sensitivity_ari.json'): fig_sensitivity(json.load(open('results/sensitivity_ari.json'))); print('fig8 ok')

# ---------- Appendix: training curves ----------
def fig_training():
    fig, ax = plt.subplots(figsize=(5, 3))
    for s in range(5):
        j = json.load(open(f'results/ae_primary_d64_s{s}.json')); lg = pd.DataFrame(j['log'])
        ax.plot(lg.epoch, np.sqrt(lg.val_mse_z) * j['metrics']['sd'], color=PAL[0], alpha=0.6, lw=0.8, label='held-out' if s == 0 else None)
        ax.plot(lg.epoch, np.sqrt(lg.train_mse_z) * j['metrics']['sd'], color=PAL[1], alpha=0.6, lw=0.8, label='development' if s == 0 else None)
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (PER units)'); ax.set_yscale('log'); ax.legend(frameon=False); save(fig, 'figA1_training')
if __name__ == '__main__':
    fig_training(); print('figA1 ok')

# ---------- Figure 9: continuous description ----------
def fig_continuous():
    c = json.load(open('results/continuous.json')); z = np.load('results/continuous_scores.npz'); S = z['S']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={'width_ratios': [1, 1, 1.3]})
    t = np.linspace(0, 1, 10); mean_curve = np.array(c['mean_curve'])
    for j, col in enumerate([PAL[0], PAL[1]]):
        L = np.array(c['loadings'][j]); sd = np.std(S[:, j])
        axes[0].plot(t, mean_curve + sd * L, '-', color=col, label=f'PC{j+1} +1 SD'); axes[0].plot(t, mean_curve - sd * L, '--', color=col, label=f'PC{j+1} $-$1 SD')
    axes[0].plot(t, mean_curve, 'k-', lw=2, label='mean'); axes[0].set_xlabel('Normalised career time'); axes[0].set_ylabel('PER'); axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    axes[0].set_title(f'PC1 {100*c["var_ratio"][0]:.0f}%, PC2 {100*c["var_ratio"][1]:.0f}% of variance', fontsize=8)
    for k in range(K):
        ii = labR == k; axes[1].scatter(S[ii, 0], S[ii, 1], s=5, color=PAL[k % len(PAL)], alpha=0.55, label=f'T{k+1}', linewidths=0)
    axes[1].set_xlabel('PC1 (career level)'); axes[1].set_ylabel('PC2 (rise vs decline)'); axes[1].legend(fontsize=6, frameon=False, ncol=4, markerscale=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    for nm in ['Michael Jordan', 'Steve Nash', 'Ralph Sampson', 'Darko Miličić', 'Adam Morrison', 'LeBron James']:
        i = meta.index[meta.name == nm]
        if len(i): i = i[0]; axes[1].annotate(nm.split()[-1], (S[i, 0], S[i, 1]), fontsize=6, xytext=(3, 3), textcoords='offset points')
    axes[2].hexbin(S[:, 0], S[:, 1], gridsize=35, cmap='Greys', mincnt=1, linewidths=0.2); axes[2].set_xlabel('PC1 (career level)'); axes[2].set_ylabel('PC2 (rise vs decline)'); axes[2].set_title('Density of players', fontsize=8)
    save(fig, 'fig9_continuous')
if __name__ == '__main__':
    fig_continuous(); print('fig9 ok')
