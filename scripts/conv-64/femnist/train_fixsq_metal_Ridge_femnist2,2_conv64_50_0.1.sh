#! /bin/bash

output='metal_femnistfix2,2_conv64_Ridge_nmaxs1qmaxtb1_SGD0.001Drop50_0.1'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='2'
mkdir -p logs
mkdir -p runs

python fed_main_fixsq.py \
--model-type conv64 \
--avg-pool False \
--projection False \
--algorithm Ridge \
--scale-factor 1. \
--learnable-scale True \
--classifier-metric euclidean \
--dataset-path fed_data/femnist/fixedsq_atleast2class1shot1query_split \
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