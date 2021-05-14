#! /bin/bash

output='metal_femnist_conv64_SVM_n20s1q5tb5_SGD0.001Drop50_0.1'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='2'
mkdir -p logs
mkdir -p runs

python fed_main.py \
--model-type conv64 \
--avg-pool False \
--projection False \
--num-classes-train 0 \
--algorithm SVM \
--scale-factor 1. \
--learnable-scale True \
--classifier-metric euclidean \
--dataset-path fed_data/femnist/at_least_20class5examples_split \
--img-side-len 28 \
--n-epochs 100 \
--batch-size-train 5 \
--n-way-train 20 \
--n-shot-train 1 \
--n-query-train 5 \
--n-iters-per-epoch 500 \
--batch-size-val 2 \
--n-way-val 20 \
--n-shot-val 1 \
--n-query-val 5 \
--do-one-shot-eval-too False \
--n-iterations-val 200 \
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