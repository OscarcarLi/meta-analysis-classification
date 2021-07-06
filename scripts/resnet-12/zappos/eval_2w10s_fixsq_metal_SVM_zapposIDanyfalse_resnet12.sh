#! /bin/bash

output='eval_metal_zapposIDanyfalse_r12_SVM_n2s10q10'
# method_dataset_model_config
device='3'
mkdir -p logs
mkdir -p runs

# we don't specify the number of ways but it is always 2.

python fed_eval_fixsq.py \
--model-type resnet_12 \
--avg-pool False \
--projection False \
--num-classes-train 0 \
--algorithm SVM \
--scale-factor 1. \
--learnable-scale True \
--classifier-metric euclidean \
--dataset-path fed_data/zappos/zappos-alltrue_vs_anyfalse-ns10-nq10 \
--novel-json novel-ID-25000.json \
--img-side-len 84 \
--batch-size-val 8 \
--preload-train False \
--eps 0. \
--checkpoint runs/metal_zapposID11000anyfalser12_SVM_n2s10q10b4_nep60_SGD0.01Drop30_0.06/chkpt_060.pt \
--output-folder ${output} \
--device-number ${device} \
--log-interval 50 