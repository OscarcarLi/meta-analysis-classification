#! /bin/bash

output='metal_tiered_r12_SVM_n5s15q6tb8_SGD0.1Drop204050'
device='1'
mkdir -p logs
mkdir -p runs

python eval.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool False \
--projection False \
--num-classes-train 0 \
--algorithm SVM \
--scale-factor 1. \
--learnable-scale True \
--dataset-path datasets/filelists/tieredImagenet-base \
--img-side-len 84 \
--batch-size-val 8 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 1250 \
--preload-train True \
--eps 0. \
--checkpoint runs/metal_tiered_r12_SVM_n5s15q6tb8_SGD0.1Drop204050/chkpt_060.pt \
--output-folder ${output} \
--device-number ${device} \
--log-interval 100
