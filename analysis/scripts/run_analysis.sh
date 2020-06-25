#!/bin/bash
export PYTHONPATH='.'

python analysis/main_analysis.py \
--algorithm SVM \
--model-type resnet \
--add-bias True \
--no-fc-layer True \
--checkpoint train_dir/classical_miniimagenet/classical_resnet_400.pt \
--n-aux-objective-steps 5 \
--num-classes 16 \
--label-offset 64 \
--optimizer adam \
--lr 0.001 \
--grad-clip 0. \
--dataset-path data/filelists/miniImagenet \
--train-aug \
--img-side-len 84 \
--batch-size-val 10 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 100 \
--output-folder analyse_SVM_with_classical_backbone \
--device-number 0,1,2,3 \
--log-interval 200 \
--eval