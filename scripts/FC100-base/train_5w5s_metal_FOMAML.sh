#! /bin/bash

output='metal_FC100-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.0005det'
# method_dataset_model_innerAlg_config_outerOpt 
device='2'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python main.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 5 \
--classifier-type linear \
--algorithm InitBasedAlgorithm \
--init-meta-algorithm FOMAML \
--inner-update-method sgd \
--alpha 0.01 \
--num-updates-inner-train 2 \
--num-updates-inner-val 20 \
--classifier-metric euclidean \
--dataset-path datasets/filelists/FC100-base \
--img-side-len 32 \
--n-epochs 60 \
--batch-size-train 4 \
--n-way-train 5 \
--n-shot-train 5 \
--n-query-train 15 \
--n-iters-per-epoch 1000 \
--batch-size-val 1 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 500 \
--support-aug True \
--query-aug True \
--randomize-query False \
--preload-train True \
--optimizer-type Adam \
--lr 0.0005 \
--weight-decay 0.0 \
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

