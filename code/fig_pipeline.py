import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
def box(x, y, w, h, text, fc='#eef3fa', fs=7.5, ec='#4a6fa5'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=fc, ec=ec, lw=1)); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, color='#333'))
box(0.1, 2.9, 1.7, 1.1, 'Basketball-Reference\nadvanced tables\n1979-80 to 2025-26\n(scraped 4 Sep 2026)', fc='#f7f7f7', ec='#888')
box(2.1, 2.9, 1.9, 1.1, 'Sample construction\ndebut 1979-80..2010-11\n>=100 min per valid season\n>=42 games, >=2 seasons', fc='#f7f7f7', ec='#888')
box(4.3, 2.9, 1.9, 1.1, 'Calendar-axis PER\ntrajectories with\nobserved-indicator\n(gaps masked)', fc='#f7f7f7', ec='#888')
box(6.5, 2.9, 1.5, 1.1, 'Split by player\n80% development\n20% held-out', fc='#fff4e0', ec='#c98a2e')
box(8.3, 2.9, 1.6, 1.1, 'External variables\nposition, draft, honours,\nWS, BPM (validation only)', fc='#f0f7ee', ec='#5a9a5a')
for x0, x1 in [(1.8, 2.1), (4.0, 4.3), (6.2, 6.5)]: arrow(x0, 3.45, x1, 3.45)
# representations row
box(0.1, 1.4, 2.2, 1.0, 'Transformer autoencoder\nd in {8,16,32,64,128}, 3-5 seeds\nheld-out RMSE vs baselines (RQ1)')
box(2.55, 1.4, 1.6, 1.0, 'Summary statistics\n(9 features)')
box(4.35, 1.4, 1.6, 1.0, 'Resampled trajectory\n(10 points + length)\nand its PCA')
box(6.15, 1.4, 1.5, 1.0, 'DTW on raw\nsequences')
box(7.9, 1.4, 2.0, 1.0, 'Single-run UMAP+HDBSCAN\n720-config grid, DBCV rule,\nseed stability (RQ2)', fc='#fbeaea', ec='#b04a4a')
for x in [1.2, 3.35, 5.15, 6.9, 8.9]: arrow(x, 2.9, x, 2.42)
# consensus row
box(0.1, 0.1, 6.5, 0.85, 'Common consensus protocol: 80% subsamples x seeds x (AE seeds), k-means k=2..12, co-association, average-linkage consensus;\nrule: max consensus Silhouette (k>=3); PAC, stability, R2-shape, external validity (RQ3, RQ4)')
box(6.9, 0.1, 3.0, 0.85, 'Structure tests: gap statistic, dip test,\nGaussian and copula null references;\nsensitivity: specs, gaps, BPM, d (RQ5)', fc='#f0f7ee', ec='#5a9a5a')
for x in [1.2, 3.35, 5.15, 6.9]: arrow(x, 1.4, min(x, 6.4), 0.97)
arrow(8.9, 1.4, 8.4, 0.97)
fig.savefig('paper/figures/fig2_pipeline.png', dpi=300, bbox_inches='tight'); fig.savefig('paper/figures/fig2_pipeline.pdf', bbox_inches='tight')
