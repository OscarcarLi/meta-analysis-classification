#! /bin/bash
# evaluate a pair of snapshots
# pass paths to jsons for alg hyperparams. The rest of the hyperparams are specified here.
# when n-runs is 1 and n-chosen-classes=the total number of test_large classes
# this is just novel_large evaluation for the algorithm snapshot.

output='metal_mini_r12_PN_n20s5q15tb1_SGD0.1Drop204050_chkpt053_vs_metal_mini_r12_Ridge_n5s15q6tb8_SGD0.1Drop20_chkpt053'
device='0,1'

export PYTHONPATH='.'

python analysis/compute_novel_acc_variance.py \
--sample 0 \
--n-chosen-classes 20 \
--n-runs 100 \
--preload-images True \
--algorithm-hyperparams-json-1 scripts/alg_hyperparams_jsons/mini_PN.json \
--algorithm-hyperparams-json-2 scripts/alg_hyperparams_jsons/mini_Ridge.json \
--checkpoint-1 runs/metal_mini_r12_PN_n20s5q15tb1_SGD0.1Drop204050/chkpt_053.pt \
--checkpoint-2 runs/metal_mini_r12_Ridge_n5s15q6tb8_SGD0.1Drop20/chkpt_053.pt \
--dataset-path data/miniImagenet-base \
--img-side-len 84 \
--batch-size-val 20 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 1000 \
--output-folder ${output} \
--device-number ${device}