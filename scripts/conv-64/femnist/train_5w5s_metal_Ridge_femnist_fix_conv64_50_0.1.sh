#! /bin/bash

output='metal_femnistfix2,2_conv64_Ridge_nmaxs1qmaxtb1_SGD0.001Drop50_0.1'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='4'
mkdir -p logs
mkdir -p runs

python fed_main_fixsq.py \
--model-type conv64 \
--avg-pool False \
--projection False \
--num-classes-train 0 \
--algorithm Ridge \
--scale-factor 1. \
--learnable-scale True \
--classifier-metric euclidean \
--dataset-path fed_data/femnist/fixedsq_atleast2class1shot1query_split \
--img-side-len 28 \
--n-epochs 100 \
--batch-size-train 1 \
--n-way-train 62 \
--n-shot-train 1 \
--n-iters-per-epoch 2509 \
--batch-size-val 1 \
--n-way-val 62 \
--n-shot-val 1 \
--do-one-shot-eval-too False \
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