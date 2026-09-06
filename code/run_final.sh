#!/bin/bash
cd /home/claude/per
while [ ! -f logs/phase3.done ]; do sleep 30; done
for r in summary raw pca; do python3 code/consensus_kmeans.py --rep $r > logs/cons_$r.log 2>&1; done
python3 code/consensus_kmeans.py --rep summary --ward > logs/cons_summary_ward.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space latent > logs/cons_ae64_latent.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space latent --seeds 0 --draws 20 --tag ae64_latent_seed0only > logs/cons_ae64_seed0.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space latent --nostd --tag ae64_latent_nostd > logs/cons_ae64_nostd.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space umap5 > logs/cons_ae64_umap5.log 2>&1
for dmm in 8 16 32 128; do python3 code/consensus_kmeans.py --rep ae$dmm --space latent > logs/cons_ae${dmm}_latent.log 2>&1; done
for sp in original2025 mp250 min3seasons nogames debut1980_2005 primary_bpm consecutive; do python3 code/consensus_kmeans.py --rep ae64 --space latent --spec $sp > logs/cons_ae64_$sp.log 2>&1; done
python3 code/consensus_kmeans.py --rep ae64 --space umap5 --clusterer hdbscan > logs/cons_ae64_umap5_hdbscan.log 2>&1
python3 code/consensus_kmeans.py --rep dtw --draws 4 > logs/cons_dtw.log 2>&1
python3 code/evaluate.py > logs/evaluate.log 2>&1
echo DONE > logs/final.done
