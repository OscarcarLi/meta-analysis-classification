#! /bin/bash

output='metal_zappos3000allfalse_conv64_Ridge_n2s10q10b1_nep100_SGD0.001Drop50_0.1'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1'
mkdir -p logs
mkdir -p runs

# we don't specify the number of ways but it is always 2.

python fed_main_fixsq.py \
--model-type conv64 \
--avg-pool False \
--projection False \
--num-classes-train 0 \
--algorithm Ridge \
--scale-factor 1. \
--learnable-scale True \
--classifier-metric euclidean \
--dataset-path fed_data/zappos/zappos-minImg1000-alltrue_vs_allfalse-nsamp8000-ns10-nq10-tr1000val1000test1000 \
--img-side-len 28 \
--n-epochs 100 \
--batch-size-train 1 \
--batch-size-val 1 \
--preload-train False \
--optimizer-type SGDM \
--lr 0.001 \
--weight-decay 0.01 \
--grad-clip 0. \
--drop-lr-epoch 50 \
--drop-factors 0.1 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device-number ${device} \
--log-interval 100 

# --fix-support 0 \
# --support-aug False \
# --query-aug True \
# --randomize-query True \

# Train SVM and Ridge first. 
# increasing the batch size for SVM, PN and Ridge [from 1 to ]
# we might need to increase the validation and testing tasks, currently it is 1000
# optimization schedule and initial learning rate. 