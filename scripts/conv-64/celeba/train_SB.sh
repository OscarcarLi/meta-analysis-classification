#! /bin/bash

output='SB_celeba_conv64_bs128_SGD0.1Drop90'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python fed_main.py \
--model-type conv64 \
--avg-pool True \
--projection False \
--algorithm SupervisedBaseline \
--num-classes-train 2 \
--classifier-type linear \
--dataset-path fed_data/celeba \
--img-side-len 84 \
--n-epochs 100 \
--batch-size-train 128 \
--val-frequency 5 \
--batch-size-val 128 \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 90 \
--drop-factors 0.1 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device-number ${device} \
--log-interval 100 > logs/${output}_train.log &
tail -f logs/${output}_train.log