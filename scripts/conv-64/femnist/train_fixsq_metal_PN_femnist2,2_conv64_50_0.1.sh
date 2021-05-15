#! /bin/bash

output='metal_femnistfix2,2_conv64_PN_nmaxs1qmaxtb1_SGD0.001Drop50_0.1'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1'
mkdir -p logs
mkdir -p runs

python fed_main_fixsq.py \
--model-type conv64 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path fed_data/femnist/fixedsq_atleast2class1shot1query_split \
--img-side-len 28 \
--n-epochs 50 \
--batch-size-train 1 \
--batch-size-val 1 \
--do-one-shot-eval-too False \
--preload-train False \
--optimizer-type SGDM \
--lr 0.001 \
--weight-decay 0.01 \
--grad-clip 0. \
--drop-lr-epoch 25 \
--drop-factors 0.2 \
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