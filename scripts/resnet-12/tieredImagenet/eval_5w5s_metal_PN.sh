#! /bin/bash

output='metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050'
device='0'
mkdir -p logs
mkdir -p runs

python eval.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/tieredImagenet-base \
--img-side-len 84 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 1000 \
--preload-train True \
--eps 0. \
--output-folder ${output} \
--device-number ${device} \
--checkpoint runs/metal_tiered_r12_PN_n20s5q15tb1_SGD0.1Drop204050/chkpt_060.pt \
--log-interval 100