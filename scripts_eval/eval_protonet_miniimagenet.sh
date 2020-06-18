#!/bin/bash

CHKPT=$1 

echo "Evaluating mini-imagenet 5w1s using $CHKPT"

python main.py \
--algorithm protonet \
--model-type impregconv \
--original-conv \
--num-channels 64 \
--retain-activation True \
--use-group-norm True \
--add-bias False \
--optimizer adam \
--slow-lr 0.001 \
--optimizer-update-interval 1 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 40000 \
--num-batches-meta-test 250 \
--meta-batch-size 8 \
--num-classes-per-batch-meta-train 5 \
--num-classes-per-batch-meta-val 5 \
--num-classes-per-batch-meta-test 5 \
--num-train-samples-per-class-meta-train 1 \
--num-train-samples-per-class-meta-val 1 \
--num-train-samples-per-class-meta-test 1 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder minim_20w1s_protonet \
--device cuda \
--device-number 0 \
--log-interval 50 \
--save-interval 1000 \
--val-interval 1000 \
--verbose False \
--eval \
--checkpoint "$CHKPT"


echo "Evaluating mini-imagenet 5w5s using $CHKPT"



python main.py \
--algorithm protonet \
--model-type impregconv \
--original-conv \
--num-channels 64 \
--retain-activation True \
--use-group-norm True \
--add-bias False \
--optimizer adam \
--slow-lr 0.001 \
--optimizer-update-interval 1 \
--model-grad-clip 0. \
--dataset miniimagenet \
--num-batches-meta-train 40000 \
--num-batches-meta-test 250 \
--meta-batch-size 8 \
--num-classes-per-batch-meta-train 5 \
--num-classes-per-batch-meta-val 5 \
--num-classes-per-batch-meta-test 5 \
--num-train-samples-per-class-meta-train 5 \
--num-train-samples-per-class-meta-val 5 \
--num-train-samples-per-class-meta-test 5 \
--num-val-samples-per-class-meta-train 15 \
--num-val-samples-per-class-meta-val 15 \
--num-val-samples-per-class-meta-test 15 \
--img-side-len 84 \
--output-folder minim_20w1s_protonet \
--device cuda \
--device-number 0 \
--log-interval 50 \
--save-interval 1000 \
--val-interval 1000 \
--verbose False \
--eval \
--checkpoint "$CHKPT"