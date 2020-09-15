#! /bin/bash 
output='metal_cifar_r12_n20_s5_q15_qp15_euc_drop12'

CUDA_VISIBLE_DEVICES=2,3 nohup python main_meta_classical.py \
--algorithm Protonet \
--model-type resnet12 \
--classifier-metric euclidean \
--projection False \
--img-side-len 32 \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--dataset-path data/filelists/cifar \
--n-epochs 60 \
--drop-lr-epoch 12 \
--num-classes-train 64 \
--batch-size-train 1 \
--n-way-train 20 \
--n-shot-train 5 \
--fix-support 0 \
--n-query-train 15 \
--n-query-pool 15 \
--batch-size-val 1 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 2000 \
--output-folder ${output} \
--device cuda \
--device-number 2,3 \
--log-interval 500 \
--train-aug > ${output}.out &
tail -f ${output}.out 

# --checkpoint train_dir_2/fixS_cifar_r12_n64_s5_q320_euc/classical_resnet_118.pt \
# --restart-iter 0 \


# --checkpoint train_dir_2/fixS5_cifar_r12_n64_s5_q15_euc/classical_resnet_009.pt \
# --restart-iter 0 \
# --train-aug > ${output}.out &

# --fix-support 10 \
# 
# --checkpoint train_dir_2/fixS_euc_MI_r12_n64_s5_q320/classical_resnet_150.pt \
# --restart-iter 0
# train-aug
# --checkpoint train_dir/classical_miniimagenet_avg_classifier_5/classical_resnet_163.pt
# --restart-iter 150