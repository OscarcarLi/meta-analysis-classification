#!/bin/bash
export PYTHONPATH='.'

python analysis/main_analysis.py \
--algorithm Finetune \
--model-type resnet \
--add-bias False \
--classifier-type ortho-classifier \
--checkpoint train_dir/classical_miniimagenet_dc_bn_with_var_ortho_scale_test/classical_resnet_299.pt \
--n-epochs 0 \
--n-aux-objective-steps 50 \
--num-classes 5 \
--label-offset 64 \
--optimizer adam \
--lr 0.001 \
--grad-clip 0. \
--dataset-path data/filelists/miniImagenet \
--train-aug \
--img-side-len 224 \
--batch-size-val 3 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 3 \
--device cuda \
--device-number 0,1,2,3 \
--log-interval 20