#!/bin/bash
cd /home/claude/per
while [ ! -f logs/ae_all.done ]; do sleep 30; done
run() { python3 -c "
import torch; torch.set_num_threads(1)
import sys; sys.path.insert(0,'code')
from autoencoder import train
train('$1', $2, $3, verbose=False)
" > logs/ae_$1_d$2_s$3.log 2>&1; }
( for sp in original2025 mp250 min3seasons; do run $sp 64 0; done ) &
( for sp in nogames debut1980_2005 primary_bpm; do run $sp 64 0; done ) &
python3 code/consensus_kmeans.py --rep ae64 --space latent > logs/cons_ae64_latent.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space umap5 > logs/cons_ae64_umap5.log 2>&1
for dmm in 8 16 32 128; do python3 code/consensus_kmeans.py --rep ae$dmm --space latent > logs/cons_ae${dmm}_latent.log 2>&1; done
wait
for sp in original2025 mp250 min3seasons nogames debut1980_2005 primary_bpm; do python3 code/consensus_kmeans.py --rep ae64 --space latent --spec $sp > logs/cons_ae64_$sp.log 2>&1; done
python3 code/stability_single_run.py > logs/stability_single_run.log 2>&1
python3 code/evaluate.py > logs/evaluate.log 2>&1
echo DONE > logs/phase2.done
