#!/bin/bash
cd /home/claude/per
run() { python3 -c "
import torch; torch.set_num_threads(1)
import sys; sys.argv=['x']; sys.path.insert(0,'code')
from autoencoder import train
train('$1', $2, $3, verbose=False)
" > logs/ae_$1_d$2_s$3.log 2>&1; }
mkdir -p logs
# queue A
( for j in "primary 64 1" "primary 64 2" "primary 64 3" "primary 64 4" "primary 8 0" "primary 16 0" "primary 32 0" "primary 128 0"; do run $j; done ) &
# queue B
( for j in "primary 8 1" "primary 16 1" "primary 32 1" "primary 128 1" "primary 8 2" "primary 16 2" "primary 32 2" "primary 128 2"; do run $j; done ) &
wait
echo ALLDONE > logs/ae_all.done
