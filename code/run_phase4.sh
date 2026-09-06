#!/bin/bash
cd /home/claude/per
while [ ! -f logs/phase3.done ]; do sleep 30; done
python3 code/consensus_kmeans.py --rep dtw --draws 4 > logs/cons_dtw.log 2>&1
python3 code/evaluate.py > logs/evaluate.log 2>&1
echo DONE > logs/phase4.done
