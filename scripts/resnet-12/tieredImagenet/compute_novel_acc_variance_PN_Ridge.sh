#! /bin/bash
# evaluate a pair of snapshots
# pass paths to jsons for alg hyperparams. The rest of the hyperparams are specified here.
# when n-runs is 1 and n-chosen-classes=the total number of test_large classes
# this is just novel_large evaluation for the algorithm snapshot.

output='metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_as_chkpt050_vs_metal_tiered_r12_Ridge_n5s15q6tb8_SGD0.1Drop204050_chkpt050'
device='1,2,3,4'

export PYTHONPATH='.'

python analysis/compute_novel_acc_variance.py \
--sample 0 \
--n-chosen-classes 552 \
--n-runs 1 \
--preload-images False \
--algorithm-hyperparams-json-1 scripts/alg_hyperparams_jsons/tiered_PN.json \
--algorithm-hyperparams-json-2 scripts/alg_hyperparams_jsons/tiered_Ridge.json \
--dataset-path datasets/filelists/tieredImagenet-base \
--img-side-len 84 \
--batch-size-val 40 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 2500 \
--output-folder ${output} \
--device-number ${device}