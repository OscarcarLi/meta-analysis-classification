#! /bin/bash

output='metal_celeba_r12_PN_n2s1q5tb20_SGD0.1Drop204050'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python fed_main.py \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path fed_data/celeba \
--img-side-len 84 \
--n-epochs 60 \
--batch-size-train 20 \
--n-way-train 2 \
--n-shot-train 1 \
--n-query-train 5 \
--n-iters-per-epoch 1000 \
--batch-size-val 20 \
--n-way-val 2 \
--n-shot-val 1 \
--do-one-shot-eval-too False \
--n-query-val 10 \
--n-iterations-val 1000 \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 20,40,50 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device-number ${device} \
--log-interval 100 > logs/${output}_train.log &
tail -f logs/${output}_train.log

# --fix-support 0 \
# --support-aug False \
# --query-aug True \
# --randomize-query True \