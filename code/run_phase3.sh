#!/bin/bash
cd /home/claude/per
while [ ! -f logs/phase2.done ]; do sleep 30; done
python3 -c "
import torch; torch.set_num_threads(2)
import sys; sys.path.insert(0,'code')
from autoencoder import train
train('consecutive', 64, 0, verbose=False)" > logs/ae_consecutive_d64_s0.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space latent --spec consecutive > logs/cons_ae64_consecutive.log 2>&1
python3 code/consensus_kmeans.py --rep ae64 --space umap5 --clusterer hdbscan > logs/cons_ae64_umap5_hdbscan.log 2>&1
python3 code/evaluate.py > logs/evaluate.log 2>&1
echo DONE > logs/phase3.done
