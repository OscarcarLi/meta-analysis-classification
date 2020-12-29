#! /bin/bash

output='fixS1_MI_r12_PNeuc_n5s1q15Vtb4_SGD0.1det'
# method_dataset_model_innerAlg_config_outerOpt 
device='1'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python main.py \
--fix-support 1 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--classifier-metric euclidean \
--dataset-path datasets/filelists/miniImagenet \
--img-side-len 84 \
--n-epochs 60 \
--batch-size-train 4 \
--n-way-train 5 \
--n-shot-train 1 \
--n-query-train 15 \
--n-iters-per-epoch 1000 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 1 \
--n-query-val 15 \
--n-iterations-val 250 \
--support-aug False \
--query-aug True \
--randomize-query True \
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
--device cuda \
--device-number ${device} \
--log-interval 100 > logs/${output}.log &
tail -f logs/${output}.log

