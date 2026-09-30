#!/bin/bash
# Reproduces every result, table and figure of the paper, in the order in which they were produced.
# Run from the repository root:  bash code/run_all.sh
# Total time on 2 CPU cores: roughly 10-12 hours (autoencoders about 5 h, DTW consensus about 3 h).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs results

# 1. Data
python3 code/build_dataset.py            # sample construction for all specifications, first NBA season, draft links
python3 code/build_bpm.py
python3 code/build_consecutive.py

# 2. Autoencoders (RQ1): d in {8,16,32,128} x seeds 0-2, d = 64 x seeds 0-4, alternative specifications with seed 0
ae() { python3 -c "
import torch; torch.set_num_threads(1)
import sys; sys.path.insert(0, 'code')
from autoencoder import train
train('$1', $2, $3, verbose=False)" > logs/ae_$1_d$2_s$3.log 2>&1; }
for s in 0 1 2 3 4; do ae primary 64 $s; done
for d in 8 16 32 128; do for s in 0 1 2; do ae primary $d $s; done; done
for sp in original2025 mp250 min3seasons nogames debut1980_2005 primary_bpm consecutive; do ae $sp 64 0; done
for s in 0 1 2 3 4; do python3 -c "
import torch; torch.set_num_threads(1)
import sys; sys.path.insert(0, 'code')
from autoencoder import train
train('primary', 64, $s, verbose=False, tag='primary_d64_s${s}_3way', three_way=True)" > logs/ae_3way_s$s.log 2>&1; done

# 3. Single-run UMAP+HDBSCAN grid and its reproducibility (RQ2)
python3 code/clustering.py --ae primary_d64_s0 > logs/cluster_s0.log 2>&1
python3 code/stability_single_run.py > logs/stability_single_run.log 2>&1

# 4. Consensus clustering, about 50 members per ensemble (RQ3, RQ4)
C="python3 code/consensus_kmeans.py"
for r in summary raw pca shape; do $C --rep $r --draws 50 > logs/cons_$r.log 2>&1; done
$C --rep summary --ward --draws 50 > logs/cons_summary_ward.log 2>&1
$C --rep ae64 --space latent --draws 10 > logs/cons_ae64_latent.log 2>&1
$C --rep ae64 --space latent --seeds 0 --draws 50 --tag ae64_latent_seed0only > logs/cons_ae64_seed0.log 2>&1
$C --rep ae64 --space latent --nostd --draws 10 --tag ae64_latent_nostd > logs/cons_ae64_nostd.log 2>&1
for d in 8 16 32 128; do $C --rep ae$d --space latent --draws 17 > logs/cons_ae${d}_latent.log 2>&1; done
$C --rep ae64 --space umap5 --draws 10 > logs/cons_ae64_umap5.log 2>&1
$C --rep ae64 --space umap5 --clusterer hdbscan --draws 10 > logs/cons_ae64_umap5_hdbscan.log 2>&1
$C --rep dtw --draws 8 > logs/cons_dtw.log 2>&1

# 5. Structureless references and tests for discrete structure
for b in 0 1 2; do for r in summary raw; do
  $C --rep ${r}_null$b --draws 50 > logs/cons_${r}_null$b.log 2>&1
  $C --rep ${r}_copula$b --draws 50 > logs/cons_${r}_copula$b.log 2>&1
done; done
python3 code/structure_tests.py > logs/structure_tests.log 2>&1
python3 code/continuous.py > logs/continuous.log 2>&1
python3 code/mixture_tests.py > logs/mixture_tests.log 2>&1
python3 code/mixture_diagnostics.py > logs/mixture_diagnostics.log 2>&1
python3 code/latent_geometry.py > logs/latent_geometry.log 2>&1
python3 code/power_simulation.py 0.0,2.0,3.0 results/power_a.json > logs/power_a.log 2>&1
python3 code/power_simulation.py 1.5,2.5,4.0 results/power_b.json > logs/power_b.log 2>&1
python3 code/dip_direction_check.py > logs/dip_direction.log 2>&1

# 6. Sensitivity analyses (RQ5)
for sp in original2025 mp250 min3seasons nogames debut1980_2005 primary_bpm consecutive; do
  $C --rep ae64 --space latent --spec $sp --draws 50 > logs/cons_ae64_$sp.log 2>&1
  $C --rep summary --spec $sp --draws 50 > logs/cons_summary_$sp.log 2>&1
done

# 7. Evaluation, numbers, tables and figures
python3 code/evaluate.py > logs/evaluate.log 2>&1
python3 code/make_numbers.py
python3 code/make_figures.py
python3 code/fig_pipeline.py
