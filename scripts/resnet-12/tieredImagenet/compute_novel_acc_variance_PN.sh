#! /bin/bash
# evaluate a single snapshot
# when n-runs is 1 and n-chosen-classes=the total number of test_large classes
# this is just novel_large evaluation for the algorithm snapshot.

output='metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_15:Apr:2021:01:03:17'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1,2,3,4'

export PYTHONPATH='/home/oscarli/projects/meta-analysis-classification'

python analysis/compute_novel_acc_variance.py \
--sample 0 \
--n-chosen-classes 552 \
--n-runs 1 \
--preload-train False \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--classifier-type no-classifier \
--classifier-metric euclidean \
--num-classes-train 0 \
--algorithm-1 ProtoNet \
--scale-factor 10. \
--learnable-scale False \
--dataset-path ~/projects/meta-analysis-classification/datasets/filelists/tieredImagenet-base \
--img-side-len 84 \
--batch-size-val 40 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 25000 \
--eps 0. \
--output-folder ${output} \
--device-number ${device} \
--checkpoint-1 /home/oscarli/projects/meta-analysis-classification/runs/metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050_base_correct_15:Apr:2021:01:03:17/chkpt_051.pt

# --chosen-classes-indices-list chosen_novel_indices_20from120.txt \
# --n-chosen-classes 160 \