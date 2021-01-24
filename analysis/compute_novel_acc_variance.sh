#! /bin/bash

output='metal_cifar-fs-base_r12_PNeuc_Ridge'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'

export PYTHONPATH='..'

CUDA_VISIBLE_DEVICES="$device" python compute_novel_acc_variance.py \
--n-chosen-classes 8 \
--n-runs 100 \
--preload-train True \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--algorithm-1 ProtoNet \
--algorithm-2 ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path ../datasets/filelists/cifar-fs-base \
--img-side-len 32 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 1000 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint-1 ../runs/metal_cifar-fs-base_r12_PN_n20s5q15tb1_SGD0.1Drop204050/chkpt_060.pt \
--checkpoint-2 ../runs/metal_cifar-fs-base_r12_PN_n5s5q15tb4_SGD0.1Drop204050/chkpt_060.pt \
--log-interval 100
