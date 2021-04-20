#! /bin/bash

output='metal_tiered_r12_SVM_n5s15q6tb8_SGD0.1Drop204050'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0,1,2,3'
mkdir -p logs
mkdir -p runs

python main.py \
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
--n-epochs 60 \
--batch-size-train 8 \
--n-way-train 5 \
--n-shot-train 15 \
--n-query-train 6 \
--n-iters-per-epoch 1000 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 1000 \
--support-aug True \
--query-aug True \
--randomize-query False \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 20,40,50 \
--lr-scheduler-type deterministic \
--eps 0.1 \
--restart-iter 0 \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--log-interval 100