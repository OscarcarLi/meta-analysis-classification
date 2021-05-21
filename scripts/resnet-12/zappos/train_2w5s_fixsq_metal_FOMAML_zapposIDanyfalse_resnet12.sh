#! /bin/bash

output='metal_zapposIDanyfalse_r12_FOMAMLinnUpd5T20Vp0.01_n2s5q10b4_nep60_SGD0.01Drop30_0.06'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='3'
mkdir -p logs
mkdir -p runs

# we don't specify the number of ways but it is always 2.

python fed_main_fixsq.py \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 2 \
--algorithm InitBasedAlgorithm \
--init-meta-algorithm FOMAML \
--inner-update-method sgd \
--alpha 0.01 \
--num-updates-inner-train 5 \
--num-updates-inner-val 20 \
--classifier-metric euclidean \
--dataset-path fed_data/zappos/zappos-alltrue_vs_anyfalse-ns5-nq10 \
--base-json base-ID-1000.json \
--val-json val-ID-1000.json \
--novel-json novel-ID-100.json \
--img-side-len 84 \
--n-epochs 60 \
--batch-size-train 4 \
--batch-size-val 4 \
--preload-train False \
--optimizer-type SGDM \
--lr 0.01 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 30 \
--drop-factors 0.06 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device-number ${device} \
--log-interval 50 