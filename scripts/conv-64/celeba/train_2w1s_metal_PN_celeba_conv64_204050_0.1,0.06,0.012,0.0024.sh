#! /bin/bash

output='metal_celeba_PN_n2s1q5tb20_SGD0.1Drop204050_0.06,0.012,0.0024_conv64'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='2'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python fed_main.py \
--model-type conv64 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path fed_data/celeba \
--img-side-len 84 \
--n-epochs 100 \
--batch-size-train 20 \
--n-way-train 2 \
--n-shot-train 1 \
--n-query-train 5 \
--n-iters-per-epoch 200 \
--batch-size-val 20 \
--n-way-val 2 \
--n-shot-val 1 \
--do-one-shot-eval-too False \
--n-query-val 10 \
--n-iterations-val 200 \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.01 \
--grad-clip 0. \
--drop-lr-epoch 20,40,50 \
--drop-factors 0.06,0.012,0.0024 \
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