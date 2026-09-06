#!/bin/bash
cd /home/claude/per
( for s in 0 1 2 3 4; do python3 -c "
import torch; torch.set_num_threads(1)
import sys; sys.path.insert(0,'code')
from autoencoder import train
train('primary', 64, $s, verbose=False, tag='primary_d64_s${s}_3way', three_way=True)" > logs/ae_3way_s$s.log 2>&1; done; echo DONE > logs/threeway.done ) &
python3 code/consensus_kmeans.py --rep dtw --draws 8 > logs/cons_dtw8.log 2>&1
echo DONE > logs/dtw8.done
wait
