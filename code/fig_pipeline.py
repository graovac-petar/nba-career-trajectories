import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42; matplotlib.rcParams['ps.fonttype'] = 42; import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
def box(x, y, w, h, text, fc='#eef3fa', fs=8.2, ec='#4a6fa5'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=fc, ec=ec, lw=1)); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, color='#333', shrinkA=0, shrinkB=0))
def line(xs, ys): ax.plot(xs, ys, color='#333', lw=1)
# data row
box(0.1, 3.3, 1.7, 1.1, 'Basketball-\nReference advanced\ntables, 1979–80\nto 2025–26', fc='#f7f7f7', ec='#888')
box(2.1, 3.3, 2.2, 1.1, 'Sample construction:\nfirst season 1979–80 to\n2010–11; ≥ 100 min per\nseason; ≥ 42 games;\n≥ 2 valid seasons', fc='#f7f7f7', ec='#888')
box(4.6, 3.3, 1.8, 1.1, 'Calendar-axis PER\ntrajectories with\nobserved indicator\n(gaps masked)', fc='#f7f7f7', ec='#888')
box(6.7, 3.3, 1.5, 1.1, 'Split by player:\n80% development,\n20% held-out', fc='#fff4e0', ec='#c98a2e')
box(8.5, 3.3, 1.4, 1.1, 'External\nvariables\n(validation only)', fc='#f0f7ee', ec='#5a9a5a')
for x0, x1 in [(1.84, 2.06), (4.34, 4.56), (6.44, 6.66)]: arrow(x0, 3.85, x1, 3.85)
# distribution line from the split to every representation
centres = [1.2, 3.35, 5.15, 6.9, 8.9]
line([7.45, 7.45], [3.3, 3.0]); line([centres[0], centres[-1]], [3.0, 3.0])
for x in centres: arrow(x, 3.0, x, 2.62)
# representations row
box(0.1, 1.6, 2.2, 1.0, 'Transformer\nautoencoder, d = 8–128,\n3–5 seeds; held-out\nRMSE (RQ1)')
box(2.55, 1.6, 1.6, 1.0, 'Summary\nstatistics\n(10 features)')
box(4.35, 1.6, 1.6, 1.0, 'Resampled\ntrajectory\nand its PCA')
box(6.15, 1.6, 1.5, 1.0, 'DTW on\nPER\nsequences')
box(7.9, 1.6, 2.0, 1.0, 'Single-run\nUMAP+HDBSCAN:\n720 configurations,\nseed stability (RQ2)', fc='#fbeaea', ec='#b04a4a')
# consensus row
box(0.1, 0.1, 6.3, 1.0, 'Consensus protocol: about 50 members (80% subsamples ×\nautoencoder seeds), k-means with k = 2, …, 12; rule:\nminimum PAC for k ≥ 3; stability, trajectory shape\nand external validity (RQ3–RQ5)')
box(6.9, 0.1, 3.0, 1.0, 'Tests for discrete structure:\ngap, dip, Gaussian and copula\nreferences, calibrated mixtures,\npower simulation; sensitivity (RQ5)', fc='#f0f7ee', ec='#5a9a5a')
for x in centres[:4]: arrow(x, 1.6, min(x, 6.2), 1.12)
arrow(6.4, 0.6, 6.9, 0.6)
fig.savefig('paper/figures/fig2_pipeline.png', dpi=600, bbox_inches='tight'); fig.savefig('paper/figures/fig2_pipeline.pdf', bbox_inches='tight')
