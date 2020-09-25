#! /bin/bash 
output='fixS5_MI_r12_n64_s5_q5_qp16_bs1_euc_metaoptdataaug_drop20'
device='2,3'
randomseed=797

CUDA_VISIBLE_DEVICES="$device" nohup python main_meta_classical.py \
--random-seed $randomseed \
--algorithm Protonet \
--model-type resnet12 \
--avg-pool True \
--classifier-metric euclidean \
--projection False \
--img-side-len 84 \
--lr 0.1 \
--eps 0. \
--weight-decay 0.0005 \
--grad-clip 0. \
--dataset-path data/filelists/miniImagenet \
--n-epochs 60 \
--drop-lr-epoch 20 \
--num-classes-train 64 \
--batch-size-train 320 \
--n-way-train 64 \
--n-shot-train 5 \
--fix-support 5 \
--n-query-train 0 \
--n-query-pool 0 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 1000 \
--output-folder ${output} \
--device cuda \
--device-number "$device" \
--log-interval 500 \
--train-aug > ${output}.out &
tail -f ${output}.out


# --checkpoint train_dir_2/fixS1_cifar_r12_n5_s1_q15_qp50_bs8_euc_metaoptdataaug_sansgap_supportaug_drop20_run1/classical_resnet_020.pt \
# --restart-iter 0 \

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